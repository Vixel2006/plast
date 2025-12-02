#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // CUDA device function to get the linear index from multi-dimensional indices and strides.
    __device__ size_t cuda_get_index(const size_t* current_indices, const size_t* strides,
                                     size_t ndim);

    // CUDA device function to increment multi-dimensional indices based on the shape.
    __device__ void cuda_increment_indices(size_t* current_indices, const size_t* shape,
                                           size_t ndim);

#ifdef __cplusplus
}
#endif
