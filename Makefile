PYTHON = python3
VENV = venv
PIP = $(VENV)/bin/pip
PYTEST = $(VENV)/bin/pytest
FLAKE8 = $(VENV)/bin/flake8
BLACK = $(VENV)/bin/black
CPPLINT = cpplint
CLANG_FORMAT = clang-format
CMAKE = cmake
BUILD_DIR = build
SRC_DIR = src
INCLUDE_DIR = include
TEST_DIR = tests

C_SOURCES = $(shell find $(SRC_DIR) -name "*.c" -o -name "*.cu")
C_HEADERS = $(wildcard $(INCLUDE_DIR)/*.h)
PYTHON_TEST_FILES = $(wildcard $(TEST_DIR)/*.py)

all: prepare init build test lint style

prepare:
	@echo "--- Preparing build environment ---"
	rm -rf $(BUILD_DIR)
	rm -rf dist *.egg-info
	@echo "--- Preparation complete ---"

init:
	@echo "--- Initializing Python virtual environment and installing dependencies ---"
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install pytest flake8 black
	$(PIP) install -r requirements.txt
	@echo "--- Virtual environment setup complete ---"

build: prepare init $(BUILD_DIR)
	@echo "--- Building C project with CMake ---"
	mkdir -p $(BUILD_DIR)
	$(CMAKE) -S . -B $(BUILD_DIR)
	$(CMAKE) --build $(BUILD_DIR)
	cp $(BUILD_DIR)/_plast_cpp_core.cpython-313-x86_64-linux-gnu.so plast/
	$(PIP) install -e . --force-reinstall
	@echo "--- Build complete ---"

test:
	@echo "--- Running Python tests ---"
	$(PYTEST) $(TEST_DIR) -v
	@echo "--- Tests complete ---"

lint:
	@echo "--- Running Cpplint and Flake8 ---"
	$(CPPLINT) $(C_SOURCES) $(C_HEADERS) 2> cpplint_errors.txt || true
	$(FLAKE8) $(PYTHON_TEST_FILES) || true
	@echo "--- Linting complete ---"

style:
	@echo "--- Applying Clang-format and Black ---"
	$(CLANG_FORMAT) -i $(C_SOURCES) $(C_HEADERS)
	$(BLACK) $(PYTHON_TEST_FILES)
	@echo "--- Styling complete ---"

clean:
	@echo "--- Cleaning build artifacts and virtual environment ---"
	rm -rf $(BUILD_DIR) $(VENV) cpplint_errors.txt
	@echo "--- Clean complete ---"

.PHONY: all init build test lint style clean
