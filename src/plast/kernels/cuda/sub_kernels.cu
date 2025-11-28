#include "plast/kernels/cuda/binary_kernels.h"
#include <cuda_runtime.h>
#include <stdio.h> // For printf in error messages

// CUDA kernel for element-wise subtraction of float tensors (contiguous)
__global__ void contig_sub_kernel_float(const float* a, const float* b, float* out, const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        out[idx] = a[idx] - b[idx];
    }
}

// CUDA kernel for element-wise subtraction of int32_t tensors (contiguous)
__global__ void contig_sub_kernel_int32(const int32_t* a, const int32_t* b, int32_t* out,
                                        const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        out[idx] = a[idx] - b[idx];
    }
}

// C-compatible wrapper for float subtraction
extern "C" void plast_cuda_sub_kernel_float(float* out, const float* in1, const float* in2,
                                            size_t num_elements)
{
    int num_threads_per_block = 256;
    int num_blocks = (num_elements + num_threads_per_block - 1) / num_threads_per_block;

    contig_sub_kernel_float<<<num_blocks, num_threads_per_block>>>(in1, in2, out, num_elements);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error in plast_cuda_sub_kernel_float: %s\n", cudaGetErrorString(err));
    }
}

// C-compatible wrapper for int32_t subtraction
extern "C" void plast_cuda_sub_kernel_int32(int32_t* out, const int32_t* in1, const int32_t* in2,
                                            size_t num_elements)
{
    int num_threads_per_block = 256;
    int num_blocks = (num_elements + num_threads_per_block - 1) / num_threads_per_block;

    contig_sub_kernel_int32<<<num_blocks, num_threads_per_block>>>(in1, in2, out, num_elements);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error in plast_cuda_sub_kernel_int32: %s\n", cudaGetErrorString(err));
    }
}
