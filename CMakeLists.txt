cmake_minimum_required(VERSION 3.10)

project(plast C CXX CUDA)

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES "86") # Set to 8.6 for NVIDIA GeForce RTX 2050

# Find pybind11
set(pybind11_DIR "${CMAKE_CURRENT_SOURCE_DIR}/venv/lib/python3.13/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

# Define plast_CUDA_ENABLED if CUDA is found
if(CUDAToolkit_FOUND)
    add_compile_definitions(PLAST_CUDA_ENABLED)
endif()

# Find SLEEF (existing logic)
find_path(SLEEF_INCLUDE_DIR sleef.h
          PATHS ${CMAKE_CURRENT_SOURCE_DIR}/vendor/sleef/build/include
          NO_DEFAULT_PATH)

find_library(SLEEF_LIBRARY NAMES sleef
             PATHS ${CMAKE_CURRENT_SOURCE_DIR}/vendor/sleef/build/lib
             NO_DEFAULT_PATH STATIC)

if(SLEEF_INCLUDE_DIR AND SLEEF_LIBRARY)
    message(STATUS "Found SLEEF include: ${SLEEF_INCLUDE_DIR}")
    message(STATUS "Found SLEEF library: ${SLEEF_LIBRARY}")
else()
    message(FATAL_ERROR "Could not find SLEEF library or include directory.")
endif()

# --- Define plast C++ Core Library ---
# This will be a static library containing all the new C++ core components
add_library(plast_cpp_core_lib STATIC
    src/plast/core/device_management.cpp
    src/plast/core/shape_utils.cpp
    src/plast/graph/node.cpp
    src/plast/execution/engine.cpp
    src/plast/tensor/tensor.cpp
    src/plast/ops/binary/add.cpp
    src/plast/ops/binary/sub.cpp
    src/plast/ops/binary/mul.cpp
    src/plast/ops/binary/matmul.cpp
    src/plast/ops/unary/abs.cpp
    src/plast/ops/unary/relu.cpp
    src/plast/ops/unary/exp.cpp
    src/plast/ops/unary/log.cpp
    src/plast/ops/unary/leaky_relu.cpp
    src/plast/ops/reduction/min.cpp
    src/plast/ops/reduction/mean.cpp
    src/plast/ops/reduction/sum.cpp
    src/plast/ops/reduction/max.cpp
    src/plast/ops/movement/view.cpp 
    src/plast/ops/movement/squeeze.cpp 
    src/plast/ops/movement/unsqueeze.cpp 
    src/plast/ops/movement/transpose.cpp
    src/plast/ops/movement/expand.cpp
    src/plast/ops/movement/broadcast.cpp 
    src/plast/kernels/cuda/cuda_kernel_utils.cu
    src/plast/kernels/cuda/init_kernels.cu
    src/plast/kernels/cuda/add_kernels.cu
    src/plast/kernels/cuda/sub_kernels.cu
    src/plast/kernels/cuda/mul_kernels.cu
    src/plast/kernels/cuda/abs_kernels.cu
    src/plast/kernels/cuda/relu_kernels.cu
    src/plast/kernels/cuda/exp_kernels.cu
    src/plast/kernels/cuda/log_kernels.cu
    src/plast/kernels/cuda/leaky_relu_kernels.cu
    src/plast/kernels/cuda/matmul_kernels.cu
    src/plast/kernels/cpu/add_kernels.c
    src/plast/kernels/cpu/sub_kernels.c
    src/plast/kernels/cpu/mul_kernels.c
    src/plast/kernels/cpu/matmul_kernels.c
    src/plast/kernels/cpu/abs_kernels.c
    src/plast/kernels/cpu/relu_kernels.c
    src/plast/kernels/cpu/exp_kernels.c
    src/plast/kernels/cpu/log_kernels.c
    src/plast/kernels/cpu/leaky_relu_kernels.c
    src/plast/kernels/cpu/min_kernels.c
    src/plast/kernels/cpu/mean_kernels.c
    src/plast/kernels/cpu/sum_kernels.c
    src/plast/kernels/cpu/max_kernels.c
    src/plast/ops/init/init.cpp
)

# Set include directories for the C++ core library
target_include_directories(plast_cpp_core_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${SLEEF_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS} # Add pybind11 includes
)

# Link libraries for the C++ core library
target_link_libraries(plast_cpp_core_lib PRIVATE
    ${SLEEF_LIBRARY}
    ${CUDAToolkit_LIBRARIES}
    cudadevrt # For CUDA device runtime
)

# Set compile options for the C++ core library
target_compile_options(plast_cpp_core_lib PRIVATE -mavx2 -mfma -march=native -mno-avx512f -mno-avx512vl)
set_target_properties(plast_cpp_core_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(plast_cpp_core_lib PROPERTIES CUDA_VISIBILITY_PRESET default)
set_property(TARGET plast_cpp_core_lib PROPERTY POSITION_INDEPENDENT_CODE ON)

# --- Build the Pybind11 Python Module ---
# This creates a Python extension module named _plast_cpp_core
pybind11_add_module(_plast_cpp_core SHARED
    plast/cpp_bindings/pybind_module.cpp
)

# Link the Python module against the C++ core library
target_link_libraries(_plast_cpp_core PRIVATE -Wl,--whole-archive plast_cpp_core_lib -Wl,--no-whole-archive)

# Set output directory for the Python module
set_target_properties(_plast_cpp_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    PREFIX "" # Remove 'lib' prefix
)
