#include "plast/kernels/cuda/broadcast_kernels.h"
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

// CUDA kernel for broadcasting
template <typename T>
__global__ void broadcast_kernel_cuda(const T* input_data,
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

        // Create padded input shape and strides for easier index calculation
        size_t padded_input_shape[MAX_NDIM];
        size_t padded_input_strides[MAX_NDIM];

        size_t input_padding_offset = output_ndim - input_ndim;

        for (size_t i = 0; i < output_ndim; ++i) {
            if (i < input_padding_offset) {
                padded_input_shape[i] = 1;
                padded_input_strides[i] = 0;
            } else {
                size_t original_input_idx = i - input_padding_offset;
                padded_input_shape[i] = input_shape[original_input_idx];
                if (input_shape[original_input_idx] == 1 && output_shape[i] > 1) {
                    padded_input_strides[i] = 0;
                } else {
                    padded_input_strides[i] = input_strides[original_input_idx];
                }
            }
        }

        // Calculate the source index based on the output coordinates and padded input strides
        size_t src_linear_index = calculate_linear_index_cuda(output_coords, padded_input_strides, output_ndim);
        
        // Copy the data
        output_data[idx] = input_data[src_linear_index];
    }
}

// Wrapper function to launch the CUDA kernel
void cuda_broadcast_kernel(const void* input_data,
                           const size_t* input_shape_host,
                           const size_t* input_strides_host,
                           size_t input_ndim,
                           void* output_data,
                           const size_t* output_shape_host,
                           size_t output_ndim,
                           size_t item_size) {
    size_t total_output_elements = 1;
    for (size_t i = 0; i < output_ndim; ++i) {
        total_output_elements *= output_shape_host[i];
    }

    // Determine grid and block dimensions
    int blockSize = 256;
    int gridSize = (total_output_elements + blockSize - 1) / blockSize;

    // Allocate device memory for shape and strides
    size_t *input_shape_dev, *input_strides_dev, *output_shape_dev;

    PLAST_CUDA_CHECK(cudaMalloc(&input_shape_dev, input_ndim * sizeof(size_t)));
    PLAST_CUDA_CHECK(cudaMalloc(&input_strides_dev, input_ndim * sizeof(size_t)));
    PLAST_CUDA_CHECK(cudaMalloc(&output_shape_dev, output_ndim * sizeof(size_t)));

    // Copy host data to device
    PLAST_CUDA_CHECK(cudaMemcpy(input_shape_dev, input_shape_host, input_ndim * sizeof(size_t), cudaMemcpyHostToDevice));
    PLAST_CUDA_CHECK(cudaMemcpy(input_strides_dev, input_strides_host, input_ndim * sizeof(size_t), cudaMemcpyHostToDevice));
    PLAST_CUDA_CHECK(cudaMemcpy(output_shape_dev, output_shape_host, output_ndim * sizeof(size_t), cudaMemcpyHostToDevice));

    // Launch kernel based on item_size (dtype)
    if (item_size == sizeof(float)) {
        broadcast_kernel_cuda<<<gridSize, blockSize>>>((const float*)input_data, input_shape_dev, input_strides_dev, input_ndim, (float*)output_data, output_shape_dev, output_ndim);
    } else if (item_size == sizeof(double)) {
        broadcast_kernel_cuda<<<gridSize, blockSize>>>((const double*)input_data, input_shape_dev, input_strides_dev, input_ndim, (double*)output_data, output_shape_dev, output_ndim);
    } else if (item_size == sizeof(int8_t)) {
        broadcast_kernel_cuda<<<gridSize, blockSize>>>((const int8_t*)input_data, input_shape_dev, input_strides_dev, input_ndim, (int8_t*)output_data, output_shape_dev, output_ndim);
    } else if (item_size == sizeof(int16_t)) {
        broadcast_kernel_cuda<<<gridSize, blockSize>>>((const int16_t*)input_data, input_shape_dev, input_strides_dev, input_ndim, (int16_t*)output_data, output_shape_dev, output_ndim);
    } else if (item_size == sizeof(int32_t)) {
        broadcast_kernel_cuda<<<gridSize, blockSize>>>((const int32_t*)input_data, input_shape_dev, input_strides_dev, input_ndim, (int32_t*)output_data, output_shape_dev, output_ndim);
    } else if (item_size == sizeof(int64_t)) {
        broadcast_kernel_cuda<<<gridSize, blockSize>>>((const int64_t*)input_data, input_shape_dev, input_strides_dev, input_ndim, (int64_t*)output_data, output_shape_dev, output_ndim);
    } else if (item_size == sizeof(uint8_t)) {
        broadcast_kernel_cuda<<<gridSize, blockSize>>>((const uint8_t*)input_data, input_shape_dev, input_strides_dev, input_ndim, (uint8_t*)output_data, output_shape_dev, output_ndim);
    } else if (item_size == sizeof(uint16_t)) {
        broadcast_kernel_cuda<<<gridSize, blockSize>>>((const uint16_t*)input_data, input_shape_dev, input_strides_dev, input_ndim, (uint16_t*)output_data, output_shape_dev, output_ndim);
    } else if (item_size == sizeof(uint32_t)) {
        broadcast_kernel_cuda<<<gridSize, blockSize>>>((const uint32_t*)input_data, input_shape_dev, input_strides_dev, input_ndim, (uint32_t*)output_data, output_shape_dev, output_ndim);
    } else if (item_size == sizeof(uint64_t)) {
        broadcast_kernel_cuda<<<gridSize, blockSize>>>((const uint64_t*)input_data, input_shape_dev, input_strides_dev, input_ndim, (uint64_t*)output_data, output_shape_dev, output_ndim);
    } else if (item_size == sizeof(bool)) {
        broadcast_kernel_cuda<<<gridSize, blockSize>>>((const bool*)input_data, input_shape_dev, input_strides_dev, input_ndim, (bool*)output_data, output_shape_dev, output_ndim);
    } else {
        // Handle unsupported data types
        // This should ideally be caught earlier or handled more gracefully
    }
    PLAST_CUDA_CHECK(cudaGetLastError());

    // Free device memory
    PLAST_CUDA_CHECK(cudaFree(input_shape_dev));
    PLAST_CUDA_CHECK(cudaFree(input_strides_dev));
    PLAST_CUDA_CHECK(cudaFree(output_shape_dev));
}
