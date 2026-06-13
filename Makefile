# ── Compilers ──
CC   = cc
NVCC = nvcc
AR   = ar

# ── Flags ──
CFLAGS  = -O3 -Wall -march=native -fopenmp
NVFLAGS = -std=c++20 -enable-tile -O3 -G -arch=sm_80
DEBUG   = -g

# ── MINIMAL build (inference only, no optimizers/scheduler) ──
ifeq ($(MINIMAL),1)
  CFLAGS += -DPLAST_MINIMAL
endif

# Sources
C_SOURCES  = $(wildcard src/core/*.c) $(wildcard src/kernels/cpu/*.c) $(wildcard src/optimizers/cpu/*.c) $(wildcard src/scheduler/*.c) $(wildcard src/data/*.c) $(wildcard *.c)
CU_SOURCES = $(wildcard src/core/*.cu) $(wildcard src/kernels/cuda/*.cu) $(wildcard src/optimizers/cuda/*.cu)

# ── CUDA ──
ifeq ($(CUDA),1)
  CFLAGS += -DCUDA_AVAILABLE
  CUDA_INCLUDES = -I/usr/local/cuda/include
  INCLUDES = -I./include -I./include/kernels/cuda -I./include/optimizers -I./include/scheduler $(CUDA_INCLUDES)
else
  INCLUDES = -I./include -I./include/optimizers -I./include/scheduler
endif

# ── Source files ──
CORE_SOURCES  = $(wildcard src/core/*.c)
CPU_KERNELS   = $(wildcard src/kernels/cpu/*.c)
CPU_OPTIMS    = $(wildcard src/optimizers/cpu/*.c)
SCHEDULER_SRC = $(wildcard src/scheduler/*.c)
API_SOURCES   = $(wildcard src/api/*.c)
DATA_SRC      = $(wildcard src/data/*.c)
CUDA_SOURCES  = $(wildcard src/kernels/cuda/*.cu) $(wildcard src/optimizers/cuda/*.cu) $(wildcard src/core/*.cu)

# MINIMAL build excludes scheduler and optimizers
ifeq ($(MINIMAL),1)
  EXCLUDED_SCHED = src/scheduler/scheduler.c src/scheduler/jit.c src/scheduler/fusion.c
  EXCLUDED_OPTIMS = src/optimizers/cpu/adam.c src/optimizers/cpu/adamw.c
  SCHEDULER_SRC := $(filter-out $(EXCLUDED_SCHED), $(SCHEDULER_SRC))
  CPU_OPTIMS := $(filter-out $(EXCLUDED_OPTIMS), $(CPU_OPTIMS))
endif

# Library sources (CPU-only)
LIB_C_SOURCES = $(CORE_SOURCES) $(CPU_KERNELS) $(CPU_OPTIMS) $(SCHEDULER_SRC) $(API_SOURCES) $(DATA_SRC)

# Object files
LIB_C_OBJS  = $(patsubst %.c, %.c.o, $(LIB_C_SOURCES))
CU_OBJS     = $(patsubst %.cu, %.cu.o, $(CUDA_SOURCES))
ALL_C_OBJS  = $(patsubst %.c, %.c.o, $(C_SOURCES) $(MAIN_SRC))

# ── Target ──
TARGET    = plastc
LIB_STATIC = libplast.a
LIB_SHARED = libplast.so

# ── Phony ──
.PHONY: all full install test test-fast test-all clean format format-c format-py help \
        lib examples install-lib install-headers

# ── Default ──
all: lib

full: all install

# ── Build static library ──
lib: $(LIB_STATIC)

$(LIB_STATIC): $(LIB_C_OBJS)
	$(AR) rcs $@ $^
	@echo "  built $@"

# ── Build shared library (CUDA only) ──
ifeq ($(CUDA),1)
SHARED_OBJS = $(LIB_C_OBJS) $(CU_OBJS)
$(LIB_SHARED): $(SHARED_OBJS)
	$(NVCC) -shared $(NVFLAGS) $(INCLUDES) $^ -o $@ -lgomp -lcudart
	@echo "  built $@"
endif

# ── Build standalone binary (existing) ──
MAIN_SRC = main.c
C_SOURCES = $(CORE_SOURCES) $(CPU_KERNELS) $(CPU_OPTIMS) $(SCHEDULER_SRC) $(DATA_SRC)

$(TARGET): $(ALL_C_OBJS) $(CU_OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $^ -o $@ -lgomp

# ── Compile rules ──
%.c.o: %.c
	$(CC) $(CFLAGS) $(DEBUG) $(INCLUDES) -c $< -o $@

%.cu.o: %.cu
	$(NVCC) $(NVFLAGS) $(DEBUG) $(INCLUDES) -c $< -o $@

# ── Examples ──
EXAMPLES = $(basename $(notdir $(wildcard examples/*.c)))
EXAMPLE_TARGETS = $(addprefix examples/,$(EXAMPLES))

examples: $(EXAMPLE_TARGETS)

examples/%: examples/%.c $(LIB_STATIC)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< -L. -lplast -lm -lgomp -fopenmp
	@echo "  built $@"

# ── Install ──
PREFIX = /usr/local
LIBDIR = $(PREFIX)/lib
INCDIR = $(PREFIX)/include/plast

install-headers:
	install -d $(INCDIR)
	install -m 644 include/plast/*.h $(INCDIR)/
	cp -r include/core $(PREFIX)/include/
	cp -r include/kernels $(PREFIX)/include/
	cp -r include/optimizers $(PREFIX)/include/
	cp -r include/scheduler $(PREFIX)/include/
	cp -r include/data $(PREFIX)/include/
	@echo "  installed headers to $(INCDIR)/"

install-lib: $(LIB_STATIC)
	install -d $(LIBDIR)
	install -m 644 $(LIB_STATIC) $(LIBDIR)/
	@echo "  installed library to $(LIBDIR)/"

install: install-headers install-lib
	@echo "  plast library installed to $(PREFIX)"

# ── Python bindings (existing) ──
install-py:
	uv pip install setuptools pybind11 numpy --no-build-isolation
	uv pip install -e . --no-build-isolation

# ── Tests (existing) ──
test:
	uv run python -m pytest tests/ -v --tb=short -m "not slow"

test-fast:
	uv run python -m pytest tests/ -v --tb=short -m "not slow and not xfail"

test-all:
	uv run python -m pytest tests/ -v --tb=short

# ── Help ──
help:
	@echo "Targets:"
	@echo "  all              Build libplast.a (CPU, default)"
	@echo "  lib              Same as all"
	@echo "  CUDA=1           Build with CUDA support"
	@echo "  MINIMAL=1        Inference-only build (no optimizers/scheduler)"
	@echo "  examples         Build example programs"
	@echo "  install          Install headers and library to $(PREFIX)"
	@echo "  plastc           Build standalone binary (existing)"
	@echo "  clean            Remove object files and build artifacts"

# ── Format ──
C_FORMAT_FILES = $(shell find . \( -name '*.c' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' \) -not -path './build/*' -not -path './.venv/*')
PY_FORMAT_FILES = $(shell find . -name '*.py' -not -path './build/*' -not -path './.venv/*')

format: format-c format-py
format-c:
	clang-format -i --style=file $(C_FORMAT_FILES)
	@echo "Formatted $$(echo $(C_FORMAT_FILES) | wc -w) C/CUDA files"
format-py:
	ruff format $(PY_FORMAT_FILES)
	@echo "Formatted $$(echo $(PY_FORMAT_FILES) | wc -w) Python files"

# ── Clean ──
clean:
	rm -f $(LIB_C_OBJS) $(CU_OBJS) $(ALL_C_OBJS)
	find . -name '*.c.o' -o -name '*.cu.o' | xargs rm -f 2>/dev/null || true
	rm -f $(TARGET) $(LIB_STATIC) $(LIB_SHARED)
	rm -f $(EXAMPLE_TARGETS)
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache __pycache__
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete
	find . -name '*.pyo' -delete
