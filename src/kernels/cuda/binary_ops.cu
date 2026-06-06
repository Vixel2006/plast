#include "kernels/ops/binary.h"
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

static inline bool shapes_equal_host(const u64 *shape1, u64 ndim1, const u64 *shape2, u64 ndim2) {
  if (ndim1 != ndim2)
    return false;
  for (u64 i = 0; i < ndim1; ++i) {
    if (shape1[i] != shape2[i])
      return false;
  }
  return true;
}

#define DEFINE_CUDA_BINARY_KERNELS(op_name, OP_EXPR, GRAD_A_EXPR, GRAD_B_EXPR)                     \
  __global__ void op_name##_cuda_forward_float_contig_kernel(const float *a, const float *b,       \
                                                             float *c, u64 num_elements) {         \
    u64 idx = blockIdx.x * blockDim.x + threadIdx.x;                                               \
    if (idx < num_elements) {                                                                      \
      c[idx] = OP_EXPR(a[idx], b[idx]);                                                            \
    }                                                                                              \
  }                                                                                                \
  __global__ void op_name##_cuda_forward_float_non_contig_kernel(                                  \
      const float *a_data, const u64 *a_strides, const u64 *a_shape, u64 a_ndim,                   \
      const float *b_data, const u64 *b_strides, const u64 *b_shape, u64 b_ndim, float *c_data,    \
      const u64 *c_strides, const u64 *c_shape, u64 c_ndim, u64 num_elements) {                    \
    u64 idx = blockIdx.x * blockDim.x + threadIdx.x;                                               \
    if (idx < num_elements) {                                                                      \
      u64 c_coords[MAX_NDIM];                                                                      \
      cuda_linear_to_coords(idx, c_shape, c_ndim, c_coords);                                       \
      u64 a_offset = cuda_get_offset_broadcast(c_coords, c_ndim, a_strides, a_shape, a_ndim);      \
      u64 b_offset = cuda_get_offset_broadcast(c_coords, c_ndim, b_strides, b_shape, b_ndim);      \
      u64 c_offset = cuda_get_offset(c_coords, c_strides, c_ndim);                                 \
      c_data[c_offset] = OP_EXPR(a_data[a_offset], b_data[b_offset]);                              \
    }                                                                                              \
  }                                                                                                \
  __global__ void op_name##_cuda_backward_float_contig_kernel(                                     \
      const float *dout, const float *a, const float *b, float *da, float *db, u64 num_elements) { \
    u64 idx = blockIdx.x * blockDim.x + threadIdx.x;                                               \
    if (idx < num_elements) {                                                                      \
      if (da) {                                                                                    \
        da[idx] += GRAD_A_EXPR(dout[idx], a[idx], b[idx]);                                         \
      }                                                                                            \
      if (db) {                                                                                    \
        db[idx] += GRAD_B_EXPR(dout[idx], a[idx], b[idx]);                                         \
      }                                                                                            \
    }                                                                                              \
  }                                                                                                \
  __global__ void op_name##_cuda_backward_float_non_contig_kernel(                                 \
      const float *dout_data, const u64 *dout_strides, const u64 *dout_shape, u64 dout_ndim,       \
      const float *a_data, const u64 *a_strides, const u64 *a_shape, u64 a_ndim,                   \
      const float *b_data, const u64 *b_strides, const u64 *b_shape, u64 b_ndim, float *da_data,   \
      const u64 *da_strides, float *db_data, const u64 *db_strides, u64 num_elements) {            \
    u64 idx = blockIdx.x * blockDim.x + threadIdx.x;                                               \
    if (idx < num_elements) {                                                                      \
      u64 coords[MAX_NDIM];                                                                        \
      cuda_linear_to_coords(idx, dout_shape, dout_ndim, coords);                                   \
      u64 dout_offset = cuda_get_offset(coords, dout_strides, dout_ndim);                          \
      float grad = dout_data[dout_offset];                                                         \
      u64 a_offset = cuda_get_offset_broadcast(coords, dout_ndim, a_strides, a_shape, a_ndim);     \
      u64 b_offset = cuda_get_offset_broadcast(coords, dout_ndim, b_strides, b_shape, b_ndim);     \
      if (da_data) {                                                                               \
        u64 da_offset = cuda_get_offset_broadcast(coords, dout_ndim, da_strides, a_shape, a_ndim); \
        float val = GRAD_A_EXPR(grad, a_data[a_offset], b_data[b_offset]);                         \
        atomicAdd(&da_data[da_offset], val);                                                       \
      }                                                                                            \
      if (db_data) {                                                                               \
        u64 db_offset = cuda_get_offset_broadcast(coords, dout_ndim, db_strides, b_shape, b_ndim); \
        float val = GRAD_B_EXPR(grad, a_data[a_offset], b_data[b_offset]);                         \
        atomicAdd(&db_data[db_offset], val);                                                       \
      }                                                                                            \
    }                                                                                              \
  }

#define DEFINE_BINARY_CUDA_DISPATCH(op_name)                                                       \
  extern "C" void op_name##_cuda_forward(const Tensor **inputs, Tensor *output,                    \
                                         KernelParams params) {                                    \
    const Tensor *a = inputs[0];                                                                   \
    const Tensor *b = inputs[1];                                                                   \
    u64 num_elements = numel(output);                                                              \
    int block_size = CUDA_BLOCK_SIZE;                                                              \
    int grid_size = CUDA_GRID_SIZE(num_elements);                                                  \
    if (is_contiguous(a) && is_contiguous(b) && is_contiguous(output) &&                           \
        shapes_equal_host(a->shape, a->ndim, b->shape, b->ndim)) {                                 \
      DISPATCH_DTYPE_CUDA(                                                                         \
          a->dtype, op_name##_cuda_forward, _contig_kernel, grid_size, block_size,                 \
          ((const float *)a->data, (const float *)b->data, (float *)output->data, num_elements));  \
    } else {                                                                                       \
      DISPATCH_DTYPE_CUDA(                                                                         \
          a->dtype, op_name##_cuda_forward, _non_contig_kernel, grid_size, block_size,             \
          ((const float *)a->data, a->strides, a->shape, a->ndim, (const float *)b->data,          \
           b->strides, b->shape, b->ndim, (float *)output->data, output->strides, output->shape,   \
           output->ndim, num_elements));                                                           \
    }                                                                                              \
    CUDA_CHECK(cudaDeviceSynchronize());                                                           \
  }                                                                                                \
  extern "C" void op_name##_cuda_backward(Tensor **inputs, const Tensor *output,                   \
                                          KernelParams params) {                                   \
    const Tensor *a = inputs[0];                                                                   \
    const Tensor *b = inputs[1];                                                                   \
    u64 num_elements = numel(output);                                                              \
    int block_size = CUDA_BLOCK_SIZE;                                                              \
    int grid_size = CUDA_GRID_SIZE(num_elements);                                                  \
    if (is_contiguous(a) && is_contiguous(b) && is_contiguous(output) &&                           \
        shapes_equal_host(a->shape, a->ndim, b->shape, b->ndim)) {                                 \
      DISPATCH_DTYPE_CUDA(                                                                         \
          a->dtype, op_name##_cuda_backward, _contig_kernel, grid_size, block_size,                \
          ((const float *)output->grad->data, (const float *)a->data, (const float *)b->data,      \
           a->requires_grad ? (float *)a->grad->data : NULL,                                       \
           b->requires_grad ? (float *)b->grad->data : NULL, num_elements));                       \
    } else {                                                                                       \
      DISPATCH_DTYPE_CUDA(a->dtype, op_name##_cuda_backward, _non_contig_kernel, grid_size,        \
                          block_size,                                                              \
                          ((const float *)output->grad->data, output->grad->strides,               \
                           output->grad->shape, output->grad->ndim, (const float *)a->data,        \
                           a->strides, a->shape, a->ndim, (const float *)b->data, b->strides,      \
                           b->shape, b->ndim, a->requires_grad ? (float *)a->grad->data : NULL,    \
                           a->requires_grad ? a->grad->strides : NULL,                             \
                           b->requires_grad ? (float *)b->grad->data : NULL,                       \
                           b->requires_grad ? b->grad->strides : NULL, num_elements));             \
    }                                                                                              \
    CUDA_CHECK(cudaDeviceSynchronize());                                                           \
  }

// Math helpers
#define CUDA_ADD_OP(x, y) ((x) + (y))
#define CUDA_ADD_GRAD_A(grad, x, y) (grad)
#define CUDA_ADD_GRAD_B(grad, x, y) (grad)

#define CUDA_SUB_OP(x, y) ((x) - (y))
#define CUDA_SUB_GRAD_A(grad, x, y) (grad)
#define CUDA_SUB_GRAD_B(grad, x, y) (-(grad))

#define CUDA_MUL_OP(x, y) ((x) * (y))
#define CUDA_MUL_GRAD_A(grad, x, y) ((grad) * (y))
#define CUDA_MUL_GRAD_B(grad, x, y) ((grad) * (x))

#define CUDA_DIV_OP(x, y) ((x) / (y))
#define CUDA_DIV_GRAD_A(grad, x, y) ((grad) / (y))
#define CUDA_DIV_GRAD_B(grad, x, y) (-(grad) * (x) / ((y) * (y)))

DEFINE_CUDA_BINARY_KERNELS(add, CUDA_ADD_OP, CUDA_ADD_GRAD_A, CUDA_ADD_GRAD_B)
DEFINE_BINARY_CUDA_DISPATCH(add)

DEFINE_CUDA_BINARY_KERNELS(sub, CUDA_SUB_OP, CUDA_SUB_GRAD_A, CUDA_SUB_GRAD_B)
DEFINE_BINARY_CUDA_DISPATCH(sub)

DEFINE_CUDA_BINARY_KERNELS(mul, CUDA_MUL_OP, CUDA_MUL_GRAD_A, CUDA_MUL_GRAD_B)
DEFINE_BINARY_CUDA_DISPATCH(mul)

DEFINE_CUDA_BINARY_KERNELS(div, CUDA_DIV_OP, CUDA_DIV_GRAD_A, CUDA_DIV_GRAD_B)
DEFINE_BINARY_CUDA_DISPATCH(div)
