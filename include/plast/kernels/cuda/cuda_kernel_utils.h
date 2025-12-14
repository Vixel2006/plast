#pragma once

#include <stddef.h>
#include <stdio.h> // For fprintf, stderr
#include <cuda_runtime.h> // For cudaError_t, cudaGetErrorString

#define MAX_NDIM 8 // Maximum number of dimensions for tensors

// Macro for CUDA error checking
#define PLAST_CUDA_CHECK(ans)                                                                      \
    {                                                                                              \
        plast_cuda_assert((ans), __FILE__, __LINE__);                                              \
    }
inline void plast_cuda_assert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "PLAST_CUDA_CHECK: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

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
