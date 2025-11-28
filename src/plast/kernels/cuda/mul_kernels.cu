
#include <cuda_runtime.h>
#include <stdint.h> // For int32_t
#include <stdio.h>

// CUDA kernel for element-wise addition of float tensors
__global__ void mul_cuda_kernel_float_kernel(float* out, const float* in1, const float* in2,
                                             size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements)
    {
        out[idx] = in1[idx] * in2[idx];
    }
}

// Wrapper function to launch the CUDA kernel for float
extern "C" void plast_cuda_mul_kernel_float(float* out, const float* in1, const float* in2,
                                            size_t num_elements)
{
    int blockSize = 256;
    int numBlocks = (num_elements + blockSize - 1) / blockSize;
    mul_cuda_kernel_float_kernel<<<numBlocks, blockSize>>>(out, in1, in2, num_elements);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error in plast_cuda_mul_kernel_float: %s\n", cudaGetErrorString(err));
    }
}

// CUDA kernel for element-wise addition of int32_t tensors
__global__ void mul_cuda_kernel_int32_kernel(int32_t* out, const int32_t* in1, const int32_t* in2,
                                             size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements)
    {
        out[idx] = in1[idx] * in2[idx];
    }
}

// Wrapper function to launch the CUDA kernel for int32_t
extern "C" void plast_cuda_mul_kernel_int32(int32_t* out, const int32_t* in1, const int32_t* in2,
                                            size_t num_elements)
{
    // Define block and grid dimensions
    int blockSize = 256;
    int numBlocks = (num_elements + blockSize - 1) / blockSize;

    // Launch the kernel
    mul_cuda_kernel_int32_kernel<<<numBlocks, blockSize>>>(out, in1, in2, num_elements);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error in plast_cuda_mul_kernel_int32: %s\n", cudaGetErrorString(err));
    }
}
