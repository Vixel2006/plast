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
    src/plast/graph/node.cpp
    src/plast/execution/engine.cpp
    src/plast/tensor/tensor.cpp
    src/plast/ops/binary/add.cpp
    src/plast/ops/binary/sub.cpp
    src/plast/kernels/cpu/add_kernels.c
    src/plast/kernels/cpu/sub_kernels.c
    src/plast/kernels/cuda/sub_kernels.cu
    src/plast/kernels/cuda/add_kernels.cu
    src/plast/kernels/cuda/init_kernels.cu
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
target_compile_options(plast_cpp_core_lib PRIVATE -mavx2 -mfma)
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

# --- Comment out or remove the old 'plast' library definition ---
# The old 'plast' library is being replaced by the new C++ core and pybind11 module.
# For now, I'll comment it out to avoid conflicts.
# add_library(plast SHARED
#   src/tensor.c
#   src/tensor_cpu.c
#   src/tensor_cuda.cu

#   # CPU Operations
#   src/ops/cpu/binary/add.c
#   src/ops/cpu/binary/sub.c
#   src/ops/cpu/binary/mul.c
#   src/ops/cpu/binary/div.c
#   src/ops/cpu/binary/matmul.c
#   src/ops/cpu/binary/conv2d.c
#   src/ops/cpu/binary/dot.c
#   src/ops/cpu/binary/pow.c

#   src/ops/cpu/binary_scalar/add.c
#   src/ops/cpu/binary_scalar/sub.c
#   src/ops/cpu/binary_scalar/rsub.c
#   src/ops/cpu/binary_scalar/mul.c
#   src/ops/cpu/binary_scalar/div.c
#   src/ops/cpu/binary_scalar/rdiv.c
#   src/ops/cpu/binary_scalar/pow.c

#   src/ops/cpu/init/zeros.c
#   src/ops/cpu/init/ones.c
#   src/ops/cpu/init/randn.c
#   src/ops/cpu/init/uniform.c
#   src/ops/cpu/init/from_data.c
#   src/ops/cpu/init/borrow.c

#   src/ops/cpu/movement/view.c
#   src/ops/cpu/movement/unsqueeze.c
#   src/ops/cpu/movement/squeeze.c
#   src/ops/cpu/movement/transpose.c
#   src/ops/cpu/movement/expand.c
#   src/ops/cpu/movement/broadcast.c
#   src/ops/cpu/movement/concat.c

#   src/ops/cpu/reduction/sum.c
#   src/ops/cpu/reduction/mean.c
#   src/ops/cpu/reduction/max.c
#   src/ops/cpu/reduction/sum_full.c
#   src/ops/cpu/reduction/mean_full.c
#   src/ops/cpu/reduction/max_full.c

#   src/ops/cpu/unary/relu.c
#   src/ops/cpu/unary/log.c
#   src/ops/cpu/unary/exp.c
#   src/ops/cpu/unary/neg.c
#   src/ops/cpu/unary/clip.c
#   src/ops/cpu/unary/abs.c

#   # CUDA Operations
#   src/ops/cuda/binary/add.cu
#   src/ops/cuda/binary/sub.cu
#   src/ops/cuda/binary/mul.cu
#   src/ops/cuda/binary/div.cu
#   src/ops/cuda/binary/pow.cu
#   src/ops/cuda/binary/matmul.cu
#   src/ops/cuda/binary/conv2d.cu
#   src/ops/cuda/binary/dot.cu

#   src/ops/cuda/binary_scalar/add.cu
#   src/ops/cuda/binary_scalar/sub.cu
#   src/ops/cuda/binary_scalar/rsub.cu
#   src/ops/cuda/binary_scalar/mul.cu
#   src/ops/cuda/binary_scalar/div.cu
#   src/ops/cuda/binary_scalar/rdiv.cu
#   src/ops/cuda/binary_scalar/pow.cu

#   src/ops/cuda/init/from_data.cu

#   src/ops/cuda/movement/concat.cu

#   src/ops/cuda/reduction/sum.cu
#   src/ops/cuda/reduction/mean.cu
#   src/ops/cuda/reduction/max.cu
#   src/ops/cuda/reduction/sum_full.cu
#   src/ops/cuda/reduction/mean_full.cu
#   src/ops/cuda/reduction/max_full.cu

#   src/ops/cuda/unary/relu.cu
#   src/ops/cuda/unary/log.cu
#   src/ops/cuda/unary/exp.cu
#   src/ops/cuda/unary/neg.cu
#   src/ops/cuda/unary/abs.cu
#   src/ops/cuda/unary/clip.cu

#   # Remaining files (autograd, utils, optimizers)
#   src/utils.c
#   src/utils/cuda/device_management.cu
#   src/utils/cuda/indexing.cu

#   # Autograd CPU
#   src/autograd/utils/autograd_utils.c
#   src/autograd/cpu/binary/add.c
#   src/autograd/cpu/binary/sub.c
#   src/autograd/cpu/binary/rsub.c
#   src/autograd/cpu/binary/mul.c
#   src/autograd/cpu/binary/pow.c
#   src/autograd/cpu/binary/div.c
#   src/autograd/cpu/binary/rdiv.c
#   src/autograd/cpu/binary/matmul.c
#   src/autograd/cpu/binary/conv2d.c
#   src/autograd/cpu/binary/dot.c
#   src/autograd/cpu/binary/utils.c
#   src/autograd/cpu/movement/concat.c
#   src/autograd/cpu/reduction/sum.c
#   src/autograd/cpu/reduction/mean.c
#   src/autograd/cpu/reduction/max.c
#   src/autograd/cpu/reduction/sum_full.c
#   src/autograd/cpu/reduction/mean_full.c
#   src/autograd/cpu/reduction/max_full.c
#   src/autograd/cpu/unary/relu.c
#   src/autograd/cpu/unary/log.c
#   src/autograd/cpu/unary/exp.c
#   src/autograd/cpu/unary/abs.c
#   src/autograd/cpu/unary/neg.c
#   src/autograd/cpu/unary/clip.c

#   # Autograd CUDA
#   src/autograd/cuda/binary/add.cu
#   src/autograd/cuda/binary/sub.cu
#   src/autograd/cuda/binary/rsub.cu
#   src/autograd/cuda/binary/mul.cu
#   src/autograd/cuda/binary/pow.cu
#   src/autograd/cuda/binary/div.cu
#   src/autograd/cuda/binary/rdiv.cu
#   src/autograd/cuda/binary/matmul.cu
#   src/autograd/cuda/binary/conv2d.cu
#   src/autograd/cuda/binary/dot.cu
#   src/autograd/cuda/movement/concat.cu
#   src/autograd/cuda/reduction/sum.cu
#   src/autograd/cuda/reduction/mean.cu
#   src/autograd/cuda/reduction/max.cu
#   src/autograd/cuda/reduction/sum_full.cu
#   src/autograd/cuda/reduction/mean_full.cu
#   src/autograd/cuda/reduction/max_full.cu
#   src/autograd/cuda/unary/relu.cu
#   src/autograd/cuda/unary/log.cu
#   src/autograd/cuda/unary/exp.cu
#   src/autograd/cuda/unary/abs.cu
#   src/autograd/cuda/unary/neg.cu
#   src/autograd/cuda/unary/clip.cu

#   # Optimizers
#   src/optimizers/cpu/sgd.c
#   src/optimizers/cpu/adam.c
#   src/optimizers/cpu/zero_grad.c
#   src/optimizers/cuda/sgd_cuda.cu
#   src/optimizers/cuda/adam_cuda.cu
#   src/optimizers/cuda/zero_grad_cuda.cu
# )

# target_include_directories(plast PUBLIC include ${SLEEF_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS} ${CMAKE_BINARY_DIR})
# target_link_libraries(plast PRIVATE ${SLEEF_LIBRARY} ${CUDAToolkit_LIBRARIES} cudadevrt)
# target_compile_options(plast PRIVATE -mavx2 -mfma)
# set_target_properties(plast PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(plast PROPERTIES CUDA_VISIBILITY_PRESET default)

# # Set output directory for the shared library
# # This will place libidrak.so (or idrak.dll/idrak.dylib) in the build directory
# include(GenerateExportHeader)
# set(CMAKE_GENERATE_EXPORT_HEADER_LOCATION "${CMAKE_BINARY_DIR}")
# generate_export_header(plast)

# # set_target_properties(plast PROPERTIES CXX_VISIBILITY_PRESET hidden)
# # set_target_properties(plast PROPERTIES C_VISIBILITY_PRESET hidden)
# # set_target_properties(plast PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

# set_target_properties(plast PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
