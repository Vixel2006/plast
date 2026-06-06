#include "kernels/ops/unary.h"
#include "kernels/cuda/cuda_utils.cuh"
#include "kernels/cuda/cuda_check.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_BLOCK_SIZE 256
#define CUDA_GRID_SIZE(n) (((n) + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE)

// ── DType Dispatch for CUDA ──
#define DISPATCH_DTYPE_CUDA(dtype, kernel_prefix, kernel_suffix, grid, block, args)                \
  do {                                                                                             \
    switch (dtype) {                                                                               \
    case FLOAT32:                                                                                  \
      kernel_prefix##_float##kernel_suffix<<<grid, block>>> args;                                  \
      break;                                                                                       \
    default:                                                                                       \
      fprintf(stderr, "plast: unsupported dtype %d\n", (int)(dtype));                              \
      break;                                                                                       \
    }                                                                                              \
  } while (0)

// ── host dispatch macro ──
#define DEFINE_UNARY_CUDA_DISPATCH(op_name)                                                        \
  extern "C" void op_name##_cuda_forward(const Tensor **inputs, Tensor *output,                    \
                                         KernelParams params) {                                    \
    const Tensor *a = inputs[0];                                                                   \
    u64 num_elements = numel(a);                                                                   \
    int block_size = CUDA_BLOCK_SIZE;                                                              \
    int grid_size = CUDA_GRID_SIZE(num_elements);                                                  \
    if (is_contiguous(a) && is_contiguous(output)) {                                               \
      DISPATCH_DTYPE_CUDA(a->dtype, op_name##_cuda_forward, _contig_kernel, grid_size, block_size, \
                          ((const float *)a->data, (float *)output->data, num_elements));          \
    } else {                                                                                       \
      DISPATCH_DTYPE_CUDA(a->dtype, op_name##_cuda_forward, _non_contig_kernel, grid_size,         \
                          block_size,                                                              \
                          ((const float *)a->data, a->strides, (float *)output->data,              \
                           output->strides, a->shape, a->ndim, num_elements));                     \
    }                                                                                              \
    CUDA_CHECK(cudaDeviceSynchronize());                                                           \
  }                                                                                                \
  extern "C" void op_name##_cuda_backward(Tensor **inputs, const Tensor *output,                   \
                                          KernelParams params) {                                   \
    const Tensor *a = inputs[0];                                                                   \
    u64 num_elements = numel(a);                                                                   \
    int block_size = CUDA_BLOCK_SIZE;                                                              \
    int grid_size = CUDA_GRID_SIZE(num_elements);                                                  \
    if (is_contiguous(a) && is_contiguous(output)) {                                               \
      DISPATCH_DTYPE_CUDA(a->dtype, op_name##_cuda_backward, _contig_kernel, grid_size,            \
                          block_size,                                                              \
                          ((const float *)output->grad->data, (const float *)a->data,              \
                           a->requires_grad ? (float *)a->grad->data : NULL, num_elements));       \
    } else {                                                                                       \
      DISPATCH_DTYPE_CUDA(                                                                         \
          a->dtype, op_name##_cuda_backward, _non_contig_kernel, grid_size, block_size,            \
          ((const float *)output->grad->data, output->grad->strides, (const float *)a->data,       \
           a->strides, a->requires_grad ? (float *)a->grad->data : NULL,                           \
           a->requires_grad ? a->grad->strides : NULL, a->shape, a->ndim, num_elements));          \
    }                                                                                              \
    CUDA_CHECK(cudaDeviceSynchronize());                                                           \
  }

#define DEFINE_UNARY_CUDA_DISPATCH_PARAM(op_name, param_expr)                                      \
  extern "C" void op_name##_cuda_forward(const Tensor **inputs, Tensor *output,                    \
                                         KernelParams params) {                                    \
    const Tensor *a = inputs[0];                                                                   \
    u64 num_elements = numel(a);                                                                   \
    int block_size = CUDA_BLOCK_SIZE;                                                              \
    int grid_size = CUDA_GRID_SIZE(num_elements);                                                  \
    if (is_contiguous(a) && is_contiguous(output)) {                                               \
      DISPATCH_DTYPE_CUDA(                                                                         \
          a->dtype, op_name##_cuda_forward, _contig_kernel, grid_size, block_size,                 \
          ((const float *)a->data, (float *)output->data, num_elements, param_expr));              \
    } else {                                                                                       \
      DISPATCH_DTYPE_CUDA(a->dtype, op_name##_cuda_forward, _non_contig_kernel, grid_size,         \
                          block_size,                                                              \
                          ((const float *)a->data, a->strides, (float *)output->data,              \
                           output->strides, a->shape, a->ndim, num_elements, param_expr));         \
    }                                                                                              \
    CUDA_CHECK(cudaDeviceSynchronize());                                                           \
  }                                                                                                \
  extern "C" void op_name##_cuda_backward(Tensor **inputs, const Tensor *output,                   \
                                          KernelParams params) {                                   \
    const Tensor *a = inputs[0];                                                                   \
    u64 num_elements = numel(a);                                                                   \
    int block_size = CUDA_BLOCK_SIZE;                                                              \
    int grid_size = CUDA_GRID_SIZE(num_elements);                                                  \
    if (is_contiguous(a) && is_contiguous(output)) {                                               \
      DISPATCH_DTYPE_CUDA(                                                                         \
          a->dtype, op_name##_cuda_backward, _contig_kernel, grid_size, block_size,                \
          ((const float *)output->grad->data, (const float *)a->data,                              \
           a->requires_grad ? (float *)a->grad->data : NULL, num_elements, param_expr));           \
    } else {                                                                                       \
      DISPATCH_DTYPE_CUDA(                                                                         \
          a->dtype, op_name##_cuda_backward, _non_contig_kernel, grid_size, block_size,            \
          ((const float *)output->grad->data, output->grad->strides, (const float *)a->data,       \
           a->strides, a->requires_grad ? (float *)a->grad->data : NULL,                           \
           a->requires_grad ? a->grad->strides : NULL, a->shape, a->ndim, num_elements,            \
           param_expr));                                                                           \
    }                                                                                              \
    CUDA_CHECK(cudaDeviceSynchronize());                                                           \
  }

// ── kernel generation macro ──
#define DEFINE_CUDA_UNARY_KERNELS(op_name, OP_EXPR, GRAD_EXPR)                                     \
  __global__ void op_name##_cuda_forward_float_contig_kernel(const float *a, float *c,             \
                                                             u64 num_elements) {                   \
    u64 idx = blockIdx.x * blockDim.x + threadIdx.x;                                               \
    if (idx < num_elements) {                                                                      \
      c[idx] = OP_EXPR(a[idx]);                                                                    \
    }                                                                                              \
  }                                                                                                \
  __global__ void op_name##_cuda_forward_float_non_contig_kernel(                                  \
      const float *a_data, const u64 *a_strides, float *c_data, const u64 *c_strides,              \
      const u64 *shape, u64 ndim, u64 num_elements) {                                              \
    u64 idx = blockIdx.x * blockDim.x + threadIdx.x;                                               \
    if (idx < num_elements) {                                                                      \
      u64 coords[MAX_NDIM];                                                                        \
      cuda_linear_to_coords(idx, shape, ndim, coords);                                             \
      u64 a_offset = cuda_get_offset(coords, a_strides, ndim);                                     \
      u64 c_offset = cuda_get_offset(coords, c_strides, ndim);                                     \
      c_data[c_offset] = OP_EXPR(a_data[a_offset]);                                                \
    }                                                                                              \
  }                                                                                                \
  __global__ void op_name##_cuda_backward_float_contig_kernel(const float *dout, const float *a,   \
                                                              float *da, u64 num_elements) {       \
    u64 idx = blockIdx.x * blockDim.x + threadIdx.x;                                               \
    if (idx < num_elements) {                                                                      \
      if (da) {                                                                                    \
        da[idx] += GRAD_EXPR(dout[idx], a[idx]);                                                   \
      }                                                                                            \
    }                                                                                              \
  }                                                                                                \
  __global__ void op_name##_cuda_backward_float_non_contig_kernel(                                 \
      const float *dout_data, const u64 *dout_strides, const float *a_data, const u64 *a_strides,  \
      float *da_data, const u64 *da_strides, const u64 *shape, u64 ndim, u64 num_elements) {       \
    u64 idx = blockIdx.x * blockDim.x + threadIdx.x;                                               \
    if (idx < num_elements) {                                                                      \
      u64 coords[MAX_NDIM];                                                                        \
      cuda_linear_to_coords(idx, shape, ndim, coords);                                             \
      u64 dout_offset = cuda_get_offset(coords, dout_strides, ndim);                               \
      u64 a_offset = cuda_get_offset(coords, a_strides, ndim);                                     \
      u64 da_offset = cuda_get_offset(coords, da_strides, ndim);                                   \
      if (da_data) {                                                                               \
        da_data[da_offset] += GRAD_EXPR(dout_data[dout_offset], a_data[a_offset]);                 \
      }                                                                                            \
    }                                                                                              \
  }

#define DEFINE_CUDA_UNARY_KERNELS_PARAM(op_name, OP_EXPR, GRAD_EXPR, param_type, param_name)       \
  __global__ void op_name##_cuda_forward_float_contig_kernel(                                      \
      const float *a, float *c, u64 num_elements, param_type param_name) {                         \
    u64 idx = blockIdx.x * blockDim.x + threadIdx.x;                                               \
    if (idx < num_elements) {                                                                      \
      c[idx] = OP_EXPR(a[idx], param_name);                                                        \
    }                                                                                              \
  }                                                                                                \
  __global__ void op_name##_cuda_forward_float_non_contig_kernel(                                  \
      const float *a_data, const u64 *a_strides, float *c_data, const u64 *c_strides,              \
      const u64 *shape, u64 ndim, u64 num_elements, param_type param_name) {                       \
    u64 idx = blockIdx.x * blockDim.x + threadIdx.x;                                               \
    if (idx < num_elements) {                                                                      \
      u64 coords[MAX_NDIM];                                                                        \
      cuda_linear_to_coords(idx, shape, ndim, coords);                                             \
      u64 a_offset = cuda_get_offset(coords, a_strides, ndim);                                     \
      u64 c_offset = cuda_get_offset(coords, c_strides, ndim);                                     \
      c_data[c_offset] = OP_EXPR(a_data[a_offset], param_name);                                    \
    }                                                                                              \
  }                                                                                                \
  __global__ void op_name##_cuda_backward_float_contig_kernel(                                     \
      const float *dout, const float *a, float *da, u64 num_elements, param_type param_name) {     \
    u64 idx = blockIdx.x * blockDim.x + threadIdx.x;                                               \
    if (idx < num_elements) {                                                                      \
      if (da) {                                                                                    \
        da[idx] += GRAD_EXPR(dout[idx], a[idx], param_name);                                       \
      }                                                                                            \
    }                                                                                              \
  }                                                                                                \
  __global__ void op_name##_cuda_backward_float_non_contig_kernel(                                 \
      const float *dout_data, const u64 *dout_strides, const float *a_data, const u64 *a_strides,  \
      float *da_data, const u64 *da_strides, const u64 *shape, u64 ndim, u64 num_elements,         \
      param_type param_name) {                                                                     \
    u64 idx = blockIdx.x * blockDim.x + threadIdx.x;                                               \
    if (idx < num_elements) {                                                                      \
      u64 coords[MAX_NDIM];                                                                        \
      cuda_linear_to_coords(idx, shape, ndim, coords);                                             \
      u64 dout_offset = cuda_get_offset(coords, dout_strides, ndim);                               \
      u64 a_offset = cuda_get_offset(coords, a_strides, ndim);                                     \
      u64 da_offset = cuda_get_offset(coords, da_strides, ndim);                                   \
      if (da_data) {                                                                               \
        da_data[da_offset] += GRAD_EXPR(dout_data[dout_offset], a_data[a_offset], param_name);     \
      }                                                                                            \
    }                                                                                              \
  }

// Math helpers
#define CUDA_OP_SIN(x) __sinf(x)
#define CUDA_GRAD_SIN(dout, x) ((dout) * __cosf(x))

#define CUDA_OP_COS(x) __cosf(x)
#define CUDA_GRAD_COS(dout, x) ((dout) * (-__sinf(x)))

#define CUDA_OP_TAN(x) __tanf(x)
#define CUDA_GRAD_TAN(dout, x) ((dout) * (1.0f + __tanf(x) * __tanf(x)))

#define CUDA_OP_EXP(x) __expf(x)
#define CUDA_GRAD_EXP(dout, x) ((dout) * __expf(x))

#define CUDA_OP_LOG(x) __logf(x)
#define CUDA_GRAD_LOG(dout, x) ((dout) / (x))

#define CUDA_OP_NEG(x) (-(x))
#define CUDA_GRAD_NEG(dout, x) (-(dout))

#define CUDA_OP_ABS(x) fabsf(x)
#define CUDA_GRAD_ABS(dout, x) (((x) > 0) ? (dout) : (((x) < 0) ? -(dout) : 0.0f))

#define CUDA_OP_LEAKY_RELU(x, alpha) (((x) > 0) ? (x) : ((x) * (alpha)))
#define CUDA_GRAD_LEAKY_RELU(dout, x, alpha) (((x) > 0) ? (dout) : ((dout) * (alpha)))

// Generate kernels and host functions
DEFINE_CUDA_UNARY_KERNELS(sin, CUDA_OP_SIN, CUDA_GRAD_SIN)
DEFINE_UNARY_CUDA_DISPATCH(sin)

DEFINE_CUDA_UNARY_KERNELS(cos, CUDA_OP_COS, CUDA_GRAD_COS)
DEFINE_UNARY_CUDA_DISPATCH(cos)

DEFINE_CUDA_UNARY_KERNELS(tan, CUDA_OP_TAN, CUDA_GRAD_TAN)
DEFINE_UNARY_CUDA_DISPATCH(tan)

DEFINE_CUDA_UNARY_KERNELS(exp, CUDA_OP_EXP, CUDA_GRAD_EXP)
DEFINE_UNARY_CUDA_DISPATCH(exp)

DEFINE_CUDA_UNARY_KERNELS(log, CUDA_OP_LOG, CUDA_GRAD_LOG)
DEFINE_UNARY_CUDA_DISPATCH(log)

DEFINE_CUDA_UNARY_KERNELS(neg, CUDA_OP_NEG, CUDA_GRAD_NEG)
DEFINE_UNARY_CUDA_DISPATCH(neg)

DEFINE_CUDA_UNARY_KERNELS(abs, CUDA_OP_ABS, CUDA_GRAD_ABS)
DEFINE_UNARY_CUDA_DISPATCH(abs)

DEFINE_CUDA_UNARY_KERNELS_PARAM(leaky_relu, CUDA_OP_LEAKY_RELU, CUDA_GRAD_LEAKY_RELU, float, alpha)
DEFINE_UNARY_CUDA_DISPATCH_PARAM(leaky_relu, params.fval)
