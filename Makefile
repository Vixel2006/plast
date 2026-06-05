# Compilers
CC   = cc
NVCC = nvcc

# Flags
CFLAGS  = -O3 -Wall -march=native -fopenmp -DCUDA_AVAILABLE
NVFLAGS = -O3 -G -arch=sm_80
DEBUG = -g

# includes
INCLUDES = -I./include -I./include/kernels/cuda -I/usr/local/cuda/include -I./include/optimizers -I./include/scheduler

# Sources
C_SOURCES  = $(wildcard src/core/*.c) $(wildcard src/kernels/cpu/*.c) $(wildcard src/optimizers/cpu/*.c) $(wildcard *.c)
CU_SOURCES = $(wildcard src/core/*.cu) $(wildcard src/kernels/cuda/*.cu) $(wildcard src/optimizers/cuda/*.cu)

# Objects
C_OBJS  = $(patsubst %.c, %.c.o, $(C_SOURCES))
CU_OBJS = $(patsubst %.cu, %.cu.o, $(CU_SOURCES))

# Target
TARGET = plastc

.PHONY: all full install test test-fast test-all clean format format-c format-py help

all: $(TARGET)

full: $(TARGET) install

help:
	@echo "Targets:"
	@echo "  all          Build native binary ($(TARGET))"
	@echo "  full         Build native binary + install Python package"
	@echo "  install      Build and install Python package (uv pip install -e .)"
	@echo "  test         Run full pytest suite (excludes @slow)"
	@echo "  test-fast    Run only fast tests (no xfail, no slow)"
	@echo "  test-all     Run all tests including @slow"
	@echo "  clean        Remove object files and build artifacts"
	@echo "  format       Format all C/CUDA (clang-format) and Python (ruff) files (100 cols)"

# Build native binary
$(TARGET): $(C_OBJS) $(CU_OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(C_OBJS) $(CU_OBJS) -o $@ -lgomp

# Compile C files
%.c.o: %.c
	$(CC) $(CFLAGS) $(DEBUG) $(INCLUDES) -c $< -o $@

# Compile CU files — explicit loop avoids pattern-rule circular deps
$(CU_OBJS): %.cu.o: %.cu
	$(NVCC) $(NVFLAGS) $(DEBUG) $(INCLUDES) -c $< -o $@

# Build and install Python package
install:
	uv pip install -e . --no-build-isolation

# Run tests
test:
	uv run python -m pytest tests/ -v --tb=short -m "not slow"

test-fast:
	uv run python -m pytest tests/ -v --tb=short -m "not slow and not xfail"

test-all:
	uv run python -m pytest tests/ -v --tb=short

# Format all C/CUDA source and header files with clang-format
C_FORMAT_FILES = $(shell find . \( -name '*.c' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' \) -not -path './build/*' -not -path './.venv/*')
PY_FORMAT_FILES = $(shell find . -name '*.py' -not -path './build/*' -not -path './.venv/*')
format: format-c format-py
format-c:
	clang-format -i --style=file $(C_FORMAT_FILES)
	@echo "Formatted $$(echo $(C_FORMAT_FILES) | wc -w) C/CUDA files"
format-py:
	ruff format $(PY_FORMAT_FILES)
	@echo "Formatted $$(echo $(PY_FORMAT_FILES) | wc -w) Python files"

clean:
	rm -f $(C_OBJS) $(CU_OBJS)
	find . -name '*.c.o' -o -name '*.cu.o' | xargs rm -f 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache __pycache__
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete
	find . -name '*.pyo' -delete
