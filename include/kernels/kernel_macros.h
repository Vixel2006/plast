#pragma once
#include "kernels/kernel_types.h"
#include <stdio.h>

// ── DType Dispatch Macro ──
#define DISPATCH_DTYPE(dtype, kernel_prefix, kernel_suffix, args)                                  \
  do {                                                                                             \
    switch (dtype) {                                                                               \
    case FLOAT32:                                                                                  \
      kernel_prefix##_float##kernel_suffix args;                                                   \
      break;                                                                                       \
    default:                                                                                       \
      fprintf(stderr, "plast: unsupported dtype %d\n", (int)(dtype));                              \
      break;                                                                                       \
    }                                                                                              \
  } while (0)

// ── CPU Unary Dispatchers ──
#define DEFINE_UNARY_CPU_FORWARD(op_name)                                                          \
  void op_name##_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params) {         \
    const Tensor *a = inputs[0];                                                                   \
    u64 num_elements = numel(a);                                                                   \
    if (is_contiguous(a) && is_contiguous(output)) {                                               \
      DISPATCH_DTYPE(a->dtype, op_name##_cpu_forward, _contig_kernel,                              \
                     ((const float *)a->data, (float *)output->data, num_elements));               \
    } else {                                                                                       \
      DISPATCH_DTYPE(a->dtype, op_name##_cpu_forward, _non_contig_kernel,                          \
                     ((const float *)a->data, a->strides, (float *)output->data, output->strides,  \
                      a->shape, a->ndim, num_elements));                                           \
    }                                                                                              \
  }

#define DEFINE_UNARY_CPU_BACKWARD(op_name)                                                         \
  void op_name##_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params) {        \
    const Tensor *a = inputs[0];                                                                   \
    u64 num_elements = numel(a);                                                                   \
    if (is_contiguous(a) && is_contiguous(output)) {                                               \
      DISPATCH_DTYPE(a->dtype, op_name##_cpu_backward, _contig_kernel,                             \
                     ((const float *)output->grad->data, (const float *)a->data,                   \
                      a->requires_grad ? (float *)a->grad->data : NULL, num_elements));            \
    } else {                                                                                       \
      DISPATCH_DTYPE(                                                                              \
          a->dtype, op_name##_cpu_backward, _non_contig_kernel,                                    \
          ((const float *)output->grad->data, output->grad->strides, (const float *)a->data,       \
           a->strides, a->requires_grad ? (float *)a->grad->data : NULL,                           \
           a->requires_grad ? a->grad->strides : NULL, a->shape, a->ndim, num_elements));          \
    }                                                                                              \
  }

#define DEFINE_UNARY_CPU_FORWARD_PARAM(op_name, param_expr)                                        \
  void op_name##_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params) {         \
    const Tensor *a = inputs[0];                                                                   \
    u64 num_elements = numel(a);                                                                   \
    if (is_contiguous(a) && is_contiguous(output)) {                                               \
      DISPATCH_DTYPE(a->dtype, op_name##_cpu_forward, _contig_kernel,                              \
                     ((const float *)a->data, (float *)output->data, num_elements, param_expr));   \
    } else {                                                                                       \
      DISPATCH_DTYPE(a->dtype, op_name##_cpu_forward, _non_contig_kernel,                          \
                     ((const float *)a->data, a->strides, (float *)output->data, output->strides,  \
                      a->shape, a->ndim, num_elements, param_expr));                               \
    }                                                                                              \
  }

#define DEFINE_UNARY_CPU_BACKWARD_PARAM(op_name, param_expr)                                       \
  void op_name##_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params) {        \
    const Tensor *a = inputs[0];                                                                   \
    u64 num_elements = numel(a);                                                                   \
    if (is_contiguous(a) && is_contiguous(output)) {                                               \
      DISPATCH_DTYPE(a->dtype, op_name##_cpu_backward, _contig_kernel,                             \
                     ((const float *)output->grad->data, (const float *)a->data,                   \
                      a->requires_grad ? (float *)a->grad->data : NULL, num_elements,              \
                      param_expr));                                                                \
    } else {                                                                                       \
      DISPATCH_DTYPE(a->dtype, op_name##_cpu_backward, _non_contig_kernel,                         \
                     ((const float *)output->grad->data, output->grad->strides,                    \
                      (const float *)a->data, a->strides,                                          \
                      a->requires_grad ? (float *)a->grad->data : NULL,                            \
                      a->requires_grad ? a->grad->strides : NULL, a->shape, a->ndim, num_elements, \
                      param_expr));                                                                \
    }                                                                                              \
  }

// ── CPU Unary Non-Contiguous Loops ──
#define DEFINE_UNARY_NONCONTIG_FORWARD(name, OP_EXPR)                                              \
  void name(const float *a_data, const u64 *a_strides, float *c_data, const u64 *c_strides,        \
            const u64 *shape, u64 ndim, u64 num_elements) {                                        \
    u64 coords[MAX_NDIM];                                                                          \
    _Pragma("omp parallel for private(coords)") for (u64 i = 0; i < num_elements; ++i) {           \
      linear_to_coords(i, shape, ndim, coords);                                                    \
      u64 a_off = get_offset(coords, a_strides, ndim);                                             \
      u64 c_off = get_offset(coords, c_strides, ndim);                                             \
      c_data[c_off] = OP_EXPR(a_data[a_off]);                                                      \
    }                                                                                              \
  }

#define DEFINE_UNARY_NONCONTIG_BACKWARD(name, GRAD_EXPR)                                           \
  void name(const float *dout_data, const u64 *dout_strides, const float *a_data,                  \
            const u64 *a_strides, float *da_data, const u64 *da_strides, const u64 *shape,         \
            u64 ndim, u64 num_elements) {                                                          \
    u64 coords[MAX_NDIM];                                                                          \
    _Pragma("omp parallel for private(coords)") for (u64 i = 0; i < num_elements; ++i) {           \
      linear_to_coords(i, shape, ndim, coords);                                                    \
      u64 dout_off = get_offset(coords, dout_strides, ndim);                                       \
      u64 a_off = get_offset(coords, a_strides, ndim);                                             \
      u64 da_off = get_offset(coords, da_strides, ndim);                                           \
      if (da_data) {                                                                               \
        da_data[da_off] += GRAD_EXPR(dout_data[dout_off], a_data[a_off]);                          \
      }                                                                                            \
    }                                                                                              \
  }

#define DEFINE_UNARY_NONCONTIG_FORWARD_PARAM(name, OP_EXPR, param_type, param_name)                \
  void name(const float *a_data, const u64 *a_strides, float *c_data, const u64 *c_strides,        \
            const u64 *shape, u64 ndim, u64 num_elements, param_type param_name) {                 \
    u64 coords[MAX_NDIM];                                                                          \
    _Pragma("omp parallel for private(coords)") for (u64 i = 0; i < num_elements; ++i) {           \
      linear_to_coords(i, shape, ndim, coords);                                                    \
      u64 a_off = get_offset(coords, a_strides, ndim);                                             \
      u64 c_off = get_offset(coords, c_strides, ndim);                                             \
      c_data[c_off] = OP_EXPR(a_data[a_off], param_name);                                          \
    }                                                                                              \
  }

#define DEFINE_UNARY_NONCONTIG_BACKWARD_PARAM(name, GRAD_EXPR, param_type, param_name)             \
  void name(const float *dout_data, const u64 *dout_strides, const float *a_data,                  \
            const u64 *a_strides, float *da_data, const u64 *da_strides, const u64 *shape,         \
            u64 ndim, u64 num_elements, param_type param_name) {                                   \
    u64 coords[MAX_NDIM];                                                                          \
    _Pragma("omp parallel for private(coords)") for (u64 i = 0; i < num_elements; ++i) {           \
      linear_to_coords(i, shape, ndim, coords);                                                    \
      u64 dout_off = get_offset(coords, dout_strides, ndim);                                       \
      u64 a_off = get_offset(coords, a_strides, ndim);                                             \
      u64 da_off = get_offset(coords, da_strides, ndim);                                           \
      if (da_data) {                                                                               \
        da_data[da_off] += GRAD_EXPR(dout_data[dout_off], a_data[a_off], param_name);              \
      }                                                                                            \
    }                                                                                              \
  }

// ── CPU Shape Op Macro ──
#define DEFINE_SHAPE_OP(op_name, compute_fn, ...)                                                  \
  void op_name##_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params) {         \
    const Tensor *a = inputs[0];                                                                   \
    output->data = a->data;                                                                        \
    output->dtype = a->dtype;                                                                      \
    output->device = a->device;                                                                    \
    output->requires_grad = a->requires_grad;                                                      \
    output->grad = a->grad;                                                                        \
    compute_fn(__VA_ARGS__);                                                                       \
  }                                                                                                \
  void op_name##_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params) {        \
    /* Shape ops have no-op backward passes */                                                     \
  }

// ── Broadcast Offset Helper ──
static inline u64 broadcast_offset(const u64 *coords, u64 coords_ndim, const u64 *strides,
                                   const u64 *shape, u64 ndim) {
  u64 offset = 0;
  int dim_shift = (int)coords_ndim - (int)ndim;
  for (u64 j = 0; j < ndim; ++j) {
    u64 coord = coords[j + dim_shift];
    if (shape[j] == 1) {
      coord = 0;
    }
    offset += coord * strides[j];
  }
  return offset;
}

// ── Reduction Coord Mapping Helper ──
static inline void reduction_output_to_input_coords(const u64 *c_coords, u64 *a_coords, u64 a_ndim,
                                                    u64 dim, bool keepdim) {
  for (u64 j = 0; j < a_ndim; ++j) {
    if (j == dim) {
      a_coords[j] = 0;
    } else {
      if (keepdim) {
        a_coords[j] = c_coords[j];
      } else {
        if (j < dim) {
          a_coords[j] = c_coords[j];
        } else {
          a_coords[j] = c_coords[j - 1];
        }
      }
    }
  }
}

// ── CPU Binary Forward/Backward Macros ──
#define DEFINE_BINARY_CPU_FORWARD(op_name, OP_EXPR, SIMD_OP_EXPR)                                  \
  void op_name##_cpu_forward_float_contig_kernel(const float *a, const float *b, float *c,         \
                                                 u64 num_elements) {                               \
    u64 i = 0;                                                                                     \
    for (; i + SIMD_WIDTH - 1 < num_elements; i += SIMD_WIDTH) {                                   \
      __m256 x = _mm256_loadu_ps(a + i);                                                           \
      __m256 y = _mm256_loadu_ps(b + i);                                                           \
      __m256 z = SIMD_OP_EXPR(x, y);                                                               \
      _mm256_storeu_ps(c + i, z);                                                                  \
    }                                                                                              \
    for (; i < num_elements; ++i) {                                                                \
      c[i] = OP_EXPR(a[i], b[i]);                                                                  \
    }                                                                                              \
  }                                                                                                \
  void op_name##_cpu_forward_float_kernel(                                                         \
      const float *a_data, const u64 *a_strides, const u64 *a_shape, u64 a_ndim,                   \
      const float *b_data, const u64 *b_strides, const u64 *b_shape, u64 b_ndim, float *c_data,    \
      const u64 *c_strides, const u64 *c_shape, u64 c_ndim) {                                      \
    u64 num_elements = 1;                                                                          \
    for (u64 i = 0; i < c_ndim; ++i) {                                                             \
      num_elements *= c_shape[i];                                                                  \
    }                                                                                              \
    u64 c_coords[MAX_NDIM];                                                                        \
    _Pragma("omp parallel for private(c_coords)") for (u64 i = 0; i < num_elements; ++i) {         \
      linear_to_coords(i, c_shape, c_ndim, c_coords);                                              \
      u64 a_offset = broadcast_offset(c_coords, c_ndim, a_strides, a_shape, a_ndim);               \
      u64 b_offset = broadcast_offset(c_coords, c_ndim, b_strides, b_shape, b_ndim);               \
      u64 c_offset = get_offset(c_coords, c_strides, c_ndim);                                      \
      c_data[c_offset] = OP_EXPR(a_data[a_offset], b_data[b_offset]);                              \
    }                                                                                              \
  }                                                                                                \
  void op_name##_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params) {         \
    const Tensor *a = inputs[0];                                                                   \
    const Tensor *b = inputs[1];                                                                   \
    switch (a->dtype) {                                                                            \
    case FLOAT32:                                                                                  \
      if (is_contiguous(a) && is_contiguous(b) && is_contiguous(output) &&                         \
          shapes_equal(a->shape, a->ndim, b->shape, b->ndim)) {                                    \
        op_name##_cpu_forward_float_contig_kernel((const float *)a->data, (const float *)b->data,  \
                                                  (float *)output->data, numel(a));                \
      } else {                                                                                     \
        op_name##_cpu_forward_float_kernel((const float *)a->data, a->strides, a->shape, a->ndim,  \
                                           (const float *)b->data, b->strides, b->shape, b->ndim,  \
                                           (float *)output->data, output->strides, output->shape,  \
                                           output->ndim);                                          \
      }                                                                                            \
      break;                                                                                       \
    default:                                                                                       \
      fprintf(stderr, "plast: unsupported dtype %d\n", (int)a->dtype);                             \
      break;                                                                                       \
    }                                                                                              \
  }

#define DEFINE_BINARY_CPU_BACKWARD(op_name, GRAD_A_EXPR, GRAD_B_EXPR, SIMD_GRAD_A_EXPR,            \
                                   SIMD_GRAD_B_EXPR)                                               \
  void op_name##_cpu_backward_float_contig_kernel(                                                 \
      const float *dout, const float *a, const float *b, float *da, float *db, u64 num_elements) { \
    u64 i = 0;                                                                                     \
    for (; i + SIMD_WIDTH - 1 < num_elements; i += SIMD_WIDTH) {                                   \
      __m256 out_grad = _mm256_loadu_ps(dout + i);                                                 \
      if (da) {                                                                                    \
        __m256 a_val = _mm256_loadu_ps(a + i);                                                     \
        __m256 b_val = _mm256_loadu_ps(b + i);                                                     \
        __m256 a_grad = _mm256_loadu_ps(da + i);                                                   \
        __m256 new_grad = _mm256_add_ps(a_grad, SIMD_GRAD_A_EXPR(out_grad, a_val, b_val));         \
        _mm256_storeu_ps(da + i, new_grad);                                                        \
      }                                                                                            \
      if (db) {                                                                                    \
        __m256 a_val = _mm256_loadu_ps(a + i);                                                     \
        __m256 b_val = _mm256_loadu_ps(b + i);                                                     \
        __m256 b_grad = _mm256_loadu_ps(db + i);                                                   \
        __m256 new_grad = _mm256_add_ps(b_grad, SIMD_GRAD_B_EXPR(out_grad, a_val, b_val));         \
        _mm256_storeu_ps(db + i, new_grad);                                                        \
      }                                                                                            \
    }                                                                                              \
    for (; i < num_elements; ++i) {                                                                \
      if (da)                                                                                      \
        da[i] += GRAD_A_EXPR(dout[i], a[i], b[i]);                                                 \
      if (db)                                                                                      \
        db[i] += GRAD_B_EXPR(dout[i], a[i], b[i]);                                                 \
    }                                                                                              \
  }                                                                                                \
  void op_name##_cpu_backward_float_kernel(                                                        \
      const float *dout_data, const u64 *dout_strides, const u64 *dout_shape, u64 dout_ndim,       \
      float *da_data, const u64 *da_strides, const u64 *da_shape, u64 da_ndim, float *db_data,     \
      const u64 *db_strides, const u64 *db_shape, u64 db_ndim, const float *a_data,                \
      const u64 *a_strides, const float *b_data, const u64 *b_strides) {                           \
    u64 num_elements = 1;                                                                          \
    for (u64 i = 0; i < dout_ndim; ++i) {                                                          \
      num_elements *= dout_shape[i];                                                               \
    }                                                                                              \
    u64 dout_coords[MAX_NDIM];                                                                     \
    _Pragma("omp parallel for private(dout_coords)") for (u64 i = 0; i < num_elements; ++i) {      \
      linear_to_coords(i, dout_shape, dout_ndim, dout_coords);                                     \
      u64 dout_offset = get_offset(dout_coords, dout_strides, dout_ndim);                          \
      float grad = dout_data[dout_offset];                                                         \
      u64 a_offset = broadcast_offset(dout_coords, dout_ndim, a_strides, da_shape, da_ndim);       \
      u64 b_offset = broadcast_offset(dout_coords, dout_ndim, b_strides, db_shape, db_ndim);       \
      if (da_data) {                                                                               \
        u64 da_offset = broadcast_offset(dout_coords, dout_ndim, da_strides, da_shape, da_ndim);   \
        float val = GRAD_A_EXPR(grad, a_data[a_offset], b_data[b_offset]);                         \
        _Pragma("omp atomic") da_data[da_offset] += val;                                           \
      }                                                                                            \
      if (db_data) {                                                                               \
        u64 db_offset = broadcast_offset(dout_coords, dout_ndim, db_strides, db_shape, db_ndim);   \
        float val = GRAD_B_EXPR(grad, a_data[a_offset], b_data[b_offset]);                         \
        _Pragma("omp atomic") db_data[db_offset] += val;                                           \
      }                                                                                            \
    }                                                                                              \
  }                                                                                                \
  void op_name##_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params) {        \
    const Tensor *a = inputs[0];                                                                   \
    const Tensor *b = inputs[1];                                                                   \
    switch (a->dtype) {                                                                            \
    case FLOAT32:                                                                                  \
      if (is_contiguous(a) && is_contiguous(b) && is_contiguous(output) &&                         \
          shapes_equal(a->shape, a->ndim, b->shape, b->ndim)) {                                    \
        op_name##_cpu_backward_float_contig_kernel(                                                \
            (const float *)output->grad->data, (const float *)a->data, (const float *)b->data,     \
            a->requires_grad ? (float *)a->grad->data : NULL,                                      \
            b->requires_grad ? (float *)b->grad->data : NULL, numel(a));                           \
      } else {                                                                                     \
        op_name##_cpu_backward_float_kernel(                                                       \
            (const float *)output->grad->data, output->grad->strides, output->grad->shape,         \
            output->grad->ndim, a->requires_grad ? (float *)a->grad->data : NULL, a->strides,      \
            a->shape, a->ndim, b->requires_grad ? (float *)b->grad->data : NULL, b->strides,       \
            b->shape, b->ndim, (const float *)a->data, a->strides, (const float *)b->data,         \
            b->strides);                                                                           \
      }                                                                                            \
      break;                                                                                       \
    default:                                                                                       \
      fprintf(stderr, "plast: unsupported dtype %d\n", (int)a->dtype);                             \
      break;                                                                                       \
    }                                                                                              \
  }
