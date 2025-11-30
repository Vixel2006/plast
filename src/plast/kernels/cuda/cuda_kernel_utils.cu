#include "plast/kernels/cuda/cuda_kernel_utils.h"

// CUDA device function to get the linear index from multi-dimensional indices and strides.
__device__ size_t cuda_get_index(const size_t* current_indices, const size_t* strides, size_t ndim)
{
    size_t index = 0;
    for (size_t i = 0; i < ndim; ++i)
    {
        if (strides[i] != 0)
        {
            index += current_indices[i] * strides[i];
        }
    }
    return index;
}

// CUDA device function to increment multi-dimensional indices based on the shape.
__device__ void cuda_increment_indices(size_t* current_indices, const size_t* shape, size_t ndim)
{
    for (int i = ndim - 1; i >= 0; --i)
    {
        current_indices[i]++;
        if (current_indices[i] < shape[i])
        {
            break;
        }
        else
        {
            current_indices[i] = 0;
        }
    }
}
