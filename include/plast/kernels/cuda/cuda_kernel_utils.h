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

    void plast_cuda_strided_copy_generic(const void* input_data, const size_t* input_shape,
                                         const size_t* input_strides, size_t input_ndim,
                                         void* output_data, const size_t* output_shape,
                                         const size_t* output_strides, size_t output_ndim,
                                         size_t item_size);

#ifdef __cplusplus
}
#endif
