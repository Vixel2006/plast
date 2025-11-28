#pragma once

#include <cuda_runtime.h>
#include <stdexcept>

// Macro to check for CUDA errors
#define CHECK_CUDA_ERROR()                                                                         \
    do                                                                                             \
    {                                                                                              \
        cudaError_t err = cudaGetLastError();                                                      \
        if (err != cudaSuccess)                                                                    \
        {                                                                                          \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" +              \
                                     std::to_string(__LINE__) + ": " + cudaGetErrorString(err));   \
        }                                                                                          \
    } while (0)

namespace plast
{
namespace kernels
{
namespace cuda
{

// Function to determine optimal grid and block dimensions for CUDA kernel launch
inline void get_grid_and_block_dims(size_t num_elements, dim3& grid_dims, dim3& block_dims)
{
    // A common practice is to use 256 or 512 threads per block
    // For simplicity, let's use 256 threads per block
    const size_t threads_per_block = 256;
    block_dims = dim3(threads_per_block);
    grid_dims = dim3((num_elements + threads_per_block - 1) / threads_per_block);
}

} // namespace cuda
} // namespace kernels
} // namespace plast
