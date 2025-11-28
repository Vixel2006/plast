#include <cuda_runtime.h>
#include <curand_kernel.h>

extern "C"
{

    // Kernel to fill a tensor with zeros
    __global__ void zeros_kernel_float(float* out, size_t num_elements)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_elements)
        {
            out[idx] = 0.0f;
        }
    }

    __global__ void zeros_kernel_int(int* out, size_t num_elements)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_elements)
        {
            out[idx] = 0;
        }
    }

    // Kernel to fill a tensor with ones
    __global__ void ones_kernel_float(float* out, size_t num_elements)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_elements)
        {
            out[idx] = 1.0f;
        }
    }

    __global__ void ones_kernel_int(int* out, size_t num_elements)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_elements)
        {
            out[idx] = 1;
        }
    }

    // Kernel for normally distributed random numbers
    __global__ void randn_kernel_float(float* out, size_t num_elements, unsigned long long seed)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_elements)
        {
            curandState_t state;
            curand_init(seed, idx, 0, &state);
            out[idx] = curand_normal(&state);
        }
    }

    // Kernel for uniformly distributed random numbers
    __global__ void uniform_kernel_float(float* out, size_t num_elements, float low, float high,
                                         unsigned long long seed)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_elements)
        {
            curandState_t state;
            curand_init(seed, idx, 0, &state);
            out[idx] = low + (high - low) * curand_uniform(&state);
        }
    }

    // Host functions to launch the kernels

    void plast_cuda_zeros_float(float* out, size_t num_elements)
    {
        size_t threads = 256;
        size_t blocks = (num_elements + threads - 1) / threads;
        zeros_kernel_float<<<blocks, threads>>>(out, num_elements);
    }

    void plast_cuda_zeros_int(int* out, size_t num_elements)
    {
        size_t threads = 256;
        size_t blocks = (num_elements + threads - 1) / threads;
        zeros_kernel_int<<<blocks, threads>>>(out, num_elements);
    }

    void plast_cuda_ones_float(float* out, size_t num_elements)
    {
        size_t threads = 256;
        size_t blocks = (num_elements + threads - 1) / threads;
        ones_kernel_float<<<blocks, threads>>>(out, num_elements);
    }

    void plast_cuda_ones_int(int* out, size_t num_elements)
    {
        size_t threads = 256;
        size_t blocks = (num_elements + threads - 1) / threads;
        ones_kernel_int<<<blocks, threads>>>(out, num_elements);
    }

    void plast_cuda_randn_float(float* out, size_t num_elements, int seed)
    {
        size_t threads = 256;
        size_t blocks = (num_elements + threads - 1) / threads;
        randn_kernel_float<<<blocks, threads>>>(out, num_elements, (unsigned long long) seed);
    }

    void plast_cuda_uniform_float(float* out, size_t num_elements, float low, float high)
    {
        size_t threads = 256;
        size_t blocks = (num_elements + threads - 1) / threads;
        // A fixed seed is not ideal for uniform, but for now it's ok.
        uniform_kernel_float<<<blocks, threads>>>(out, num_elements, low, high, 1234ULL);
    }

} // extern "C"
