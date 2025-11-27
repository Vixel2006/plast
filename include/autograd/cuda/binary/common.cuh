#ifndef AUTOGRAD_CUDA_BINARY_COMMON_CUH
#define AUTOGRAD_CUDA_BINARY_COMMON_CUH

#include "autograd/autograd_binary.h"
#include "autograd/cuda/broadcast_utils.cuh"
#include "axon_export.h"
#include "cuda_utils.h" // For CHECK_CUDA
#include "logger.h"

#include "tensor.h"
#include "utils.h" // For numel
#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE_DIM 16

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

// Kernel declarations
__global__ void contig_add_grad_kernel(const float* out_grad, float* prev_grad, int n);

__global__ void noncontig_add_grad_kernel(const float* out_grad, float* prev_grad, int n,
                                          const int* prev_shape, const int* prev_strides,
                                          int prev_ndim, const int* out_shape,
                                          const int* out_strides, int out_ndim);
__global__ void noncontig_sub_grad_kernel(const float* out_grad, float* prev_grad, int n,
                                          const int* shape, const int* strides, int ndim);
__global__ void sub_grad_kernel(const float* out_grad, float* prev_grad, int n);
__global__ void mul_grad_kernel(const float* out_grad, float* prev_grad, const float* other_data, int n);
__global__ void scalar_mul_grad_kernel(const float* out_grad, float* prev_grad, float scalar,
                                       int n);
AXON_EXPORT __global__ void noncontig_mul_grad_kernel(const float* out_grad, float* prev_grad,
                                                      const float* other_data, int n,
                                                      const int* shape, const int* strides,
                                                      int ndim);
__global__ void noncontig_scalar_mul_grad_kernel(const float* out_grad, float* prev_grad,
                                                 float scalar, int n, const int* shape,
                                                 const int* strides, int ndim);
__global__ void scalar_pow_grad_kernel(const float* out_grad, float* prev_grad,
                                       float power, int n);
__global__ void base_pow_grad_kernel(const float* out_grad, float* base_data, float* base_grad,
                                     float* power_data, float* power_grad, int n);
__global__ void exponent_pow_grad_kernel(const float* out_grad, const float* out_data,
                                         const float* base_data, float* power_grad, int n);
__global__ void noncontig_scalar_pow_grad_kernel(const float* out_grad, float* prev_data,
                                                 float* prev_grad, float power, int n,
                                                 const int* shape, const int* strides, int ndim);
__global__ void noncontig_base_pow_grad_kernel(const float* out_grad, float* base_data,
                                               float* base_grad, float* power_data, int n,
                                               const int* shape, const int* strides, int ndim);
__global__ void noncontig_exponent_pow_grad_kernel(const float* out_grad, const float* out_data,
                                                   float* base_data, float* power_grad, int n,
                                                   const int* shape, const int* strides, int ndim);
__global__ void numerator_div_grad_kernel(const float* out_grad, float* prev_grad,
                                          const float* denominator, int n);
__global__ void denominator_div_grad_kernel(const float* out_grad, const float* out_data,
                                            float* prev_grad, const float* denominator, int n);
static __inline__ __global__ void noncontig_numerator_div_grad_kernel(
    const float* out_grad, float* prev_grad, const float* denominator, int n, const int* prev_shape,
    const int* prev_strides, int prev_ndim, const int* denom_shape, const int* denom_strides,
    int denom_ndim, const int* out_shape, const int* out_strides, int out_ndim)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        int out_offset =
            get_broadcasted_input_idx(i, out_shape, out_ndim, out_shape, out_strides, out_ndim);
        int prev_grad_offset =
            get_broadcasted_input_idx(i, out_shape, out_ndim, prev_shape, prev_strides, prev_ndim);
        int denom_offset = get_broadcasted_input_idx(i, out_shape, out_ndim, denom_shape,
                                                     denom_strides, denom_ndim);
        atomicAdd(&prev_grad[prev_grad_offset],
                  out_grad[out_offset] / (denominator[denom_offset] + 1e-7f));
    }
}

__global__ void noncontig_denominator_div_grad_kernel(const float* out_grad, const float* out_data,
                                                      float* prev_grad, const float* denominator, int n,
                                                      const int* prev_shape, const int* prev_strides,
                                                      int prev_ndim, const int* out_shape,
                                                      const int* out_strides, int out_ndim);
__global__ void scalar_div_grad_kernel(const float* out_grad, float* prev_grad,
                                       float scalar_denominator, int n);
__global__ void scalar_rdiv_grad_kernel(const float* out_grad, const float* out_data,
                                        float* prev_grad, float scalar_numerator,
                                        const float* prev_data, int n);
__global__ void noncontig_scalar_rdiv_grad_kernel(const float* out_grad, const float* out_data,
                                                  float* prev_grad, float scalar_numerator,
                                                  const float* prev_data, int n, const int* prev_shape,
                                                  const int* prev_strides, int prev_ndim,
                                                  const int* out_shape, const int* out_strides,
                                                  int out_ndim);
__global__ void matmul_grad_kernel(const float* A_data, const int* A_shape, const int* A_strides, int A_ndim,
                                   const float* B_data, const int* B_shape, const int* B_strides, int B_ndim,
                                   float* C_data, const int* C_shape, const int* C_strides, int C_ndim,
                                   int B_dim, int N, int P, int K, bool transpose_A, bool transpose_B);

#endif // AUTOGRAD_CUDA_BINARY_COMMON_CUH
