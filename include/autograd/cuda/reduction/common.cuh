#ifndef AXON_AUTOGRAD_CUDA_REDUCTION_COMMON_CUH
#define AXON_AUTOGRAD_CUDA_REDUCTION_COMMON_CUH

#include "logger.h"
#include "tensor.h"
#include "utils/indexing.cuh"
#include <assert.h>
#include <math.h>

const float EPSILON = 1e-5f;

#define CHECK_CUDA()                                                                               \
    do                                                                                             \
    {                                                                                              \
        cudaError_t err = cudaGetLastError();                                                      \
        if (err != cudaSuccess)                                                                    \
        {                                                                                          \
            LOG_ERROR("CUDA runtime error at %s:%d: %s", __FILE__, __LINE__,                       \
                      cudaGetErrorString(err));                                                    \
            assert(0 && "CUDA runtime error");                                                     \
        }                                                                                          \
    } while (0)

typedef struct
{
    int axis;
} ReductionExtras;

// Kernel declarations
__global__ void sum_full_grad_kernel(float* in_grad_data, const float* output_grad, int in_size,
                                     const int* in_grad_shape, const int* in_grad_strides,
                                     int in_grad_ndim);
__global__ void mean_full_grad_kernel(float* in_grad_data, const float* output_grad, int in_size,
                                      const int* in_grad_shape, const int* in_grad_strides,
                                      int in_grad_ndim);
__global__ void max_full_grad_kernel(float* in_grad_data, const float* in_data, const float* output_grad,
                                     int in_size, const float* max, const int* in_grad_shape,
                                     const int* in_grad_strides, int in_grad_ndim);
__global__ void sum_grad_kernel(const float* out_grad, float* in_grad, const int* shape, int ndim,
                                int axis, int n, const int* in_grad_shape,
                                const int* in_grad_strides, int in_grad_ndim);
__global__ void mean_grad_kernel(const float* out_grad, float* in_grad, const int* shape, int ndim,
                                 int axis, int n, const int* in_grad_shape,
                                 const int* in_grad_strides, int in_grad_ndim);
__global__ void max_grad_kernel(const float* out_grad, float* in_grad, const float* in_data,
                                const float* out_data, const int* shape, const int* in_strides,
                                const int* out_strides, int in_ndim, int out_ndim, int reduced_dim,
                                int n, const int* in_grad_shape, const int* in_grad_strides,
                                int in_grad_ndim);

#endif // AXON_AUTOGRAD_CUDA_REDUCTION_COMMON_CUH
