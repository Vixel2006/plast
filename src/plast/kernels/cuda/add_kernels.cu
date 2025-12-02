#include "plast/core/types.h"
#include "plast/kernels/cuda/cuda_kernel_utils.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

__global__ void add_cuda_kernel_float_kernel(float* out, const float* in1, const float* in2,
                                             size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements)
    {
        out[idx] = in1[idx] + in2[idx];
    }
}

extern "C" void plast_cuda_add_kernel_float(float* out, const float* in1, const float* in2,
                                            size_t num_elements)
{
    int blockSize = 256;
    int numBlocks = (num_elements + blockSize - 1) / blockSize;
    add_cuda_kernel_float_kernel<<<numBlocks, blockSize>>>(out, in1, in2, num_elements);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error in plast_cuda_add_kernel_float: %s\n", cudaGetErrorString(err));
    }
}

__global__ void add_cuda_kernel_strided_float_kernel(float* out, const float* in1, const float* in2,
                                                     const size_t* out_shape, size_t out_ndim,
                                                     const size_t* in1_strides,
                                                     const size_t* in2_strides,
                                                     size_t total_elements)
{
    size_t flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat_idx >= total_elements) return;

    // Reconstruct multi-dimensional indices from flat_idx
    size_t current_indices[MAX_TENSOR_DIMS]; // Use fixed-size array
    size_t temp_flat_idx = flat_idx;
    for (int i = out_ndim - 1; i >= 0; --i)
    {
        current_indices[i] = temp_flat_idx % out_shape[i];
        temp_flat_idx /= out_shape[i];
    }

    size_t in1_idx = cuda_get_index(current_indices, in1_strides, out_ndim);
    size_t in2_idx = cuda_get_index(current_indices, in2_strides, out_ndim);
    out[flat_idx] = in1[in1_idx] + in2[in2_idx];
}

extern "C" void plast_cuda_add_kernel_strided_float(float* out, const float* in1, const float* in2,
                                                    const size_t* out_shape, size_t out_ndim,
                                                    const size_t* in1_strides,
                                                    const size_t* in2_strides)
{
    size_t total_elements = 1;
    for (size_t i = 0; i < out_ndim; ++i)
    {
        total_elements *= out_shape[i];
    }

    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;

    add_cuda_kernel_strided_float_kernel<<<numBlocks, blockSize>>>(
        out, in1, in2, out_shape, out_ndim, in1_strides, in2_strides, total_elements);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error in plast_cuda_add_kernel_strided_float: %s\n",
                cudaGetErrorString(err));
    }
}

__global__ void add_cuda_kernel_int32_kernel(int32_t* out, const int32_t* in1, const int32_t* in2,
                                             size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements)
    {
        out[idx] = in1[idx] + in2[idx];
    }
}

extern "C" void plast_cuda_add_kernel_int32(int32_t* out, const int32_t* in1, const int32_t* in2,
                                            size_t num_elements)
{
    int blockSize = 256;
    int numBlocks = (num_elements + blockSize - 1) / blockSize;

    add_cuda_kernel_int32_kernel<<<numBlocks, blockSize>>>(out, in1, in2, num_elements);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error in plast_cuda_add_kernel_int32: %s\n", cudaGetErrorString(err));
    }
}

__global__ void add_cuda_kernel_strided_int32_kernel(int32_t* out, const int32_t* in1,
                                                     const int32_t* in2, const size_t* out_shape,
                                                     size_t out_ndim, const size_t* in1_strides,
                                                     const size_t* in2_strides,
                                                     size_t total_elements)
{
    size_t flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat_idx >= total_elements) return;

    // Reconstruct multi-dimensional indices from flat_idx
    size_t current_indices[MAX_TENSOR_DIMS]; // Use fixed-size array
    size_t temp_flat_idx = flat_idx;
    for (int i = out_ndim - 1; i >= 0; --i)
    {
        current_indices[i] = temp_flat_idx % out_shape[i];
        temp_flat_idx /= out_shape[i];
    }

    size_t in1_idx = cuda_get_index(current_indices, in1_strides, out_ndim);
    size_t in2_idx = cuda_get_index(current_indices, in2_strides, out_ndim);
    out[flat_idx] = in1[in1_idx] + in2[in2_idx];
}

extern "C" void plast_cuda_add_kernel_strided_int32(int32_t* out, const int32_t* in1,
                                                    const int32_t* in2, const size_t* out_shape,
                                                    size_t out_ndim, const size_t* in1_strides,
                                                    const size_t* in2_strides)
{
    size_t total_elements = 1;
    for (size_t i = 0; i < out_ndim; ++i)
    {
        total_elements *= out_shape[i];
    }

    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;

    add_cuda_kernel_strided_int32_kernel<<<numBlocks, blockSize>>>(
        out, in1, in2, out_shape, out_ndim, in1_strides, in2_strides, total_elements);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error in plast_cuda_add_kernel_strided_int32: %s\n",
                cudaGetErrorString(err));
    }
}
