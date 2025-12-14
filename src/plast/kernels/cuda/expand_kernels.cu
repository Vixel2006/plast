#include "plast/kernels/cuda/expand_kernels.h"
#include "plast/kernels/cuda/cuda_kernel_utils.h" // For PLAST_CUDA_CHECK and other utilities
#include <cuda_runtime.h>

// Helper to calculate the linear index in a multi-dimensional array given coordinates and strides
__device__ static size_t calculate_linear_index_cuda(const size_t* coords, const size_t* strides, size_t ndim) {
    size_t index = 0;
    for (size_t i = 0; i < ndim; ++i) {
        index += coords[i] * strides[i];
    }
    return index;
}

// CUDA kernel for expanding
template <typename T>
__global__ void expand_kernel_cuda(const T* input_data,
                                   const size_t* input_shape,
                                   const size_t* input_strides,
                                   size_t input_ndim,
                                   T* output_data,
                                   const size_t* output_shape,
                                   size_t output_ndim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    size_t total_output_elements = 1;
    for (size_t i = 0; i < output_ndim; ++i) {
        total_output_elements *= output_shape[i];
    }

    if (idx < total_output_elements) {
        // Calculate output coordinates
        size_t output_coords[MAX_NDIM]; // MAX_NDIM should be defined in cuda_kernel_utils.h
        size_t temp_idx = idx;
        for (int i = output_ndim - 1; i >= 0; --i) {
            output_coords[i] = temp_idx % output_shape[i];
            temp_idx /= output_shape[i];
        }

        // Calculate effective input strides for expansion
        size_t effective_input_strides[MAX_NDIM];
        for (size_t i = 0; i < input_ndim; ++i) {
            if (input_shape[i] == 1 && output_shape[i] > 1) {
                effective_input_strides[i] = 0; // Broadcast this dimension
            } else {
                effective_input_strides[i] = input_strides[i];
            }
        }

        // Calculate the source index based on the output coordinates and effective input strides
        size_t src_linear_index = calculate_linear_index_cuda(output_coords, effective_input_strides, input_ndim);
        
        // Copy the data
        output_data[idx] = input_data[src_linear_index];
    }
}

// Wrapper function to launch the CUDA kernel
void cuda_expand_kernel(const void* input_data,
                        const size_t* input_shape,
                        const size_t* input_strides,
                        size_t input_ndim,
                        void* output_data,
                        const size_t* output_shape,
                        size_t output_ndim,
                        size_t item_size) {
    size_t total_output_elements = 1;
    for (size_t i = 0; i < output_ndim; ++i) {
        total_output_elements *= output_shape[i];
    }

    // Determine grid and block dimensions
    int blockSize = 256;
    int gridSize = (total_output_elements + blockSize - 1) / blockSize;

    // Launch kernel based on item_size (dtype)
    if (item_size == sizeof(float)) {
        expand_kernel_cuda<<<gridSize, blockSize>>>((const float*)input_data, input_shape, input_strides, input_ndim, (float*)output_data, output_shape, output_ndim);
    } else if (item_size == sizeof(double)) {
        expand_kernel_cuda<<<gridSize, blockSize>>>((const double*)input_data, input_shape, input_strides, input_ndim, (double*)output_data, output_shape, output_ndim);
    } else if (item_size == sizeof(int8_t)) {
        expand_kernel_cuda<<<gridSize, blockSize>>>((const int8_t*)input_data, input_shape, input_strides, input_ndim, (int8_t*)output_data, output_shape, output_ndim);
    } else if (item_size == sizeof(int16_t)) {
        expand_kernel_cuda<<<gridSize, blockSize>>>((const int16_t*)input_data, input_shape, input_strides, input_ndim, (int16_t*)output_data, output_shape, output_ndim);
    } else if (item_size == sizeof(int32_t)) {
        expand_kernel_cuda<<<gridSize, blockSize>>>((const int32_t*)input_data, input_shape, input_strides, input_ndim, (int32_t*)output_data, output_shape, output_ndim);
    } else if (item_size == sizeof(int64_t)) {
        expand_kernel_cuda<<<gridSize, blockSize>>>((const int64_t*)input_data, input_shape, input_strides, input_ndim, (int64_t*)output_data, output_shape, output_ndim);
    } else if (item_size == sizeof(uint8_t)) {
        expand_kernel_cuda<<<gridSize, blockSize>>>((const uint8_t*)input_data, input_shape, input_strides, input_ndim, (uint8_t*)output_data, output_shape, output_ndim);
    } else if (item_size == sizeof(uint16_t)) {
        expand_kernel_cuda<<<gridSize, blockSize>>>((const uint16_t*)input_data, input_shape, input_strides, input_ndim, (uint16_t*)output_data, output_shape, output_ndim);
    } else if (item_size == sizeof(uint32_t)) {
        expand_kernel_cuda<<<gridSize, blockSize>>>((const uint32_t*)input_data, input_shape, input_strides, input_ndim, (uint32_t*)output_data, output_shape, output_ndim);
    } else if (item_size == sizeof(uint64_t)) {
        expand_kernel_cuda<<<gridSize, blockSize>>>((const uint64_t*)input_data, input_shape, input_strides, input_ndim, (uint64_t*)output_data, output_shape, output_ndim);
    } else if (item_size == sizeof(bool)) {
        expand_kernel_cuda<<<gridSize, blockSize>>>((const bool*)input_data, input_shape, input_strides, input_ndim, (bool*)output_data, output_shape, output_ndim);
    } else {
        // Handle unsupported data types
        // This should ideally be caught earlier or handled more gracefully
    }
    PLAST_CUDA_CHECK(cudaGetLastError());
}
