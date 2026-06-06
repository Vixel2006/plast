#include "kernels/ops/reduce.h"
#include "kernels/cpu_utils.h"
#include "kernels/kernel_macros.h"
#include <float.h>
#include <omp.h>
#include <string.h>

#define REDUCE_ADD(a, b) ((a) + (b))
#define REDUCE_MAX(a, b) ((a) > (b) ? (a) : (b))
#define REDUCE_MIN(a, b) ((a) < (b) ? (a) : (b))

#define POST_SUM(acc, n) (acc)
#define POST_MEAN(acc, n) ((acc) / (n))
#define POST_MAX(acc, n) (acc)
#define POST_MIN(acc, n) (acc)

#define SCALE_SUM(grad, n) (grad)
#define SCALE_MEAN(grad, n) ((grad) / (n))

#define COMPARE_MAX(x, val) ((x) > (val))
#define COMPARE_MIN(x, val) ((x) < (val))

// ── CPU Reduction Forward Macro ──
#define DEFINE_REDUCE_CPU_FORWARD(op_name, INIT_VAL, REDUCE_OP, POST_PROCESS)                      \
  void op_name##_cpu_forward_float_contig_kernel(const float *a, float *c, u64 num_elements) {     \
    float acc = INIT_VAL;                                                                          \
    for (u64 i = 0; i < num_elements; ++i) {                                                       \
      acc = REDUCE_OP(acc, a[i]);                                                                  \
    }                                                                                              \
    c[0] = POST_PROCESS(acc, num_elements);                                                        \
  }                                                                                                \
  void op_name##_cpu_forward_float_non_contig_kernel(const float *a_data, const u64 *a_strides,    \
                                                     const u64 *shape, u64 ndim, u64 num_elements, \
                                                     float *c_data) {                              \
    float acc = INIT_VAL;                                                                          \
    u64 coords[MAX_NDIM];                                                                          \
    for (u64 i = 0; i < num_elements; ++i) {                                                       \
      linear_to_coords(i, shape, ndim, coords);                                                    \
      u64 a_offset = get_offset(coords, a_strides, ndim);                                          \
      acc = REDUCE_OP(acc, a_data[a_offset]);                                                      \
    }                                                                                              \
    c_data[0] = POST_PROCESS(acc, num_elements);                                                   \
  }                                                                                                \
  void op_name##_cpu_forward_float_dim_kernel(                                                     \
      const float *a_data, const u64 *a_strides, const u64 *a_shape, u64 a_ndim, float *c_data,    \
      const u64 *c_strides, const u64 *c_shape, u64 c_ndim, u64 dim, bool keepdim) {               \
    u64 output_num_elements = 1;                                                                   \
    for (u64 i = 0; i < c_ndim; ++i)                                                               \
      output_num_elements *= c_shape[i];                                                           \
    _Pragma("omp parallel for") for (u64 i = 0; i < output_num_elements; ++i) {                    \
      u64 c_coords[MAX_NDIM];                                                                      \
      linear_to_coords(i, c_shape, c_ndim, c_coords);                                              \
      float acc = INIT_VAL;                                                                        \
      u64 a_coords[MAX_NDIM];                                                                      \
      reduction_output_to_input_coords(c_coords, a_coords, a_ndim, dim, keepdim);                  \
      u64 dim_len = a_shape[dim];                                                                  \
      for (u64 k = 0; k < dim_len; ++k) {                                                          \
        a_coords[dim] = k;                                                                         \
        u64 a_offset = get_offset(a_coords, a_strides, a_ndim);                                    \
        acc = REDUCE_OP(acc, a_data[a_offset]);                                                    \
      }                                                                                            \
      c_data[i] = POST_PROCESS(acc, dim_len);                                                      \
    }                                                                                              \
  }                                                                                                \
  void op_name##_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params) {         \
    const Tensor *a = inputs[0];                                                                   \
    u64 dim = params.dim;                                                                          \
    bool keepdim = params.keepdim;                                                                 \
    if (dim == MAX_NDIM + 1) {                                                                     \
      u64 num_elements = numel(a);                                                                 \
      if (is_contiguous(a)) {                                                                      \
        DISPATCH_DTYPE(a->dtype, op_name##_cpu_forward, _contig_kernel,                            \
                       ((const float *)a->data, (float *)output->data, num_elements));             \
      } else {                                                                                     \
        DISPATCH_DTYPE(a->dtype, op_name##_cpu_forward, _non_contig_kernel,                        \
                       ((const float *)a->data, a->strides, a->shape, a->ndim, num_elements,       \
                        (float *)output->data));                                                   \
      }                                                                                            \
    } else {                                                                                       \
      compute_reduction_shape_strides(a->shape, a->ndim, dim, keepdim, output->shape,              \
                                      &output->ndim, output->strides);                             \
      DISPATCH_DTYPE(a->dtype, op_name##_cpu_forward, _dim_kernel,                                 \
                     ((const float *)a->data, a->strides, a->shape, a->ndim,                       \
                      (float *)output->data, output->strides, output->shape, output->ndim, dim,    \
                      keepdim));                                                                   \
    }                                                                                              \
  }

// ── CPU Reduction Backward Macro for Sum and Mean ──
#define DEFINE_REDUCE_CPU_BACKWARD_SUM_MEAN(op_name, SCALE_EXPR)                                   \
  void op_name##_cpu_backward_float_contig_kernel(const float *dout, const float *a, float *da,    \
                                                  u64 num_elements) {                              \
    float grad = SCALE_EXPR(dout[0], num_elements);                                                \
    _Pragma("omp parallel for") for (u64 i = 0; i < num_elements; ++i) {                           \
      da[i] += grad;                                                                               \
    }                                                                                              \
  }                                                                                                \
  void op_name##_cpu_backward_float_non_contig_kernel(const float *dout_data, float *da_data,      \
                                                      const u64 *da_strides, const u64 *shape,     \
                                                      u64 ndim, u64 num_elements) {                \
    float grad = SCALE_EXPR(dout_data[0], num_elements);                                           \
    u64 coords[MAX_NDIM];                                                                          \
    _Pragma("omp parallel for private(coords)") for (u64 i = 0; i < num_elements; ++i) {           \
      linear_to_coords(i, shape, ndim, coords);                                                    \
      u64 da_offset = get_offset(coords, da_strides, ndim);                                        \
      da_data[da_offset] += grad;                                                                  \
    }                                                                                              \
  }                                                                                                \
  void op_name##_cpu_backward_float_dim_kernel(                                                    \
      const float *a_data, const u64 *a_strides, const u64 *a_shape, u64 a_ndim,                   \
      const float *c_data, const u64 *c_strides, const u64 *c_shape, u64 c_ndim,                   \
      const float *dc_data, float *da_data, const u64 *da_strides, u64 dim, bool keepdim) {        \
    u64 output_num_elements = 1;                                                                   \
    for (u64 i = 0; i < c_ndim; ++i)                                                               \
      output_num_elements *= c_shape[i];                                                           \
    _Pragma("omp parallel for") for (u64 i = 0; i < output_num_elements; ++i) {                    \
      u64 c_coords[MAX_NDIM];                                                                      \
      linear_to_coords(i, c_shape, c_ndim, c_coords);                                              \
      float grad = SCALE_EXPR(dc_data[get_offset(c_coords, c_strides, c_ndim)], a_shape[dim]);     \
      u64 a_coords[MAX_NDIM];                                                                      \
      reduction_output_to_input_coords(c_coords, a_coords, a_ndim, dim, keepdim);                  \
      u64 dim_len = a_shape[dim];                                                                  \
      for (u64 k = 0; k < dim_len; ++k) {                                                          \
        a_coords[dim] = k;                                                                         \
        u64 da_offset = get_offset(a_coords, da_strides, a_ndim);                                  \
        da_data[da_offset] += grad;                                                                \
      }                                                                                            \
    }                                                                                              \
  }                                                                                                \
  void op_name##_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params) {        \
    const Tensor *a = inputs[0];                                                                   \
    u64 dim = params.dim;                                                                          \
    bool keepdim = params.keepdim;                                                                 \
    if (a->requires_grad) {                                                                        \
      if (dim == MAX_NDIM + 1) {                                                                   \
        u64 num_elements = numel(a);                                                               \
        if (is_contiguous(a)) {                                                                    \
          DISPATCH_DTYPE(a->dtype, op_name##_cpu_backward, _contig_kernel,                         \
                         ((const float *)output->grad->data, (const float *)a->data,               \
                          (float *)a->grad->data, num_elements));                                  \
        } else {                                                                                   \
          DISPATCH_DTYPE(a->dtype, op_name##_cpu_backward, _non_contig_kernel,                     \
                         ((const float *)output->grad->data, (float *)a->grad->data,               \
                          a->grad->strides, a->shape, a->ndim, num_elements));                     \
        }                                                                                          \
      } else {                                                                                     \
        DISPATCH_DTYPE(a->dtype, op_name##_cpu_backward, _dim_kernel,                              \
                       ((const float *)a->data, a->strides, a->shape, a->ndim,                     \
                        (const float *)output->data, output->strides, output->shape, output->ndim, \
                        (const float *)output->grad->data, (float *)a->grad->data,                 \
                        a->grad->strides, dim, keepdim));                                          \
      }                                                                                            \
    }                                                                                              \
  }

// ── CPU Reduction Backward Macro for Max and Min ──
#define DEFINE_REDUCE_CPU_BACKWARD_MAX_MIN(op_name, INIT_VAL, COMPARE_OP)                          \
  void op_name##_cpu_backward_float_contig_kernel(const float *dout, const float *a, float *da,    \
                                                  u64 num_elements) {                              \
    float extreme_val = INIT_VAL;                                                                  \
    for (u64 i = 0; i < num_elements; ++i) {                                                       \
      if (COMPARE_OP(a[i], extreme_val))                                                           \
        extreme_val = a[i];                                                                        \
    }                                                                                              \
    float grad = dout[0];                                                                          \
    u64 count = 0;                                                                                 \
    for (u64 i = 0; i < num_elements; ++i) {                                                       \
      if (a[i] == extreme_val)                                                                     \
        count++;                                                                                   \
    }                                                                                              \
    if (count == 0)                                                                                \
      return;                                                                                      \
    grad /= count;                                                                                 \
    _Pragma("omp parallel for") for (u64 i = 0; i < num_elements; ++i) {                           \
      if (a[i] == extreme_val)                                                                     \
        da[i] += grad;                                                                             \
    }                                                                                              \
  }                                                                                                \
  void op_name##_cpu_backward_float_non_contig_kernel(                                             \
      const float *a_data, const u64 *a_strides, const float *c_data, const float *dc_data,        \
      float *da_data, const u64 *da_strides, const u64 *shape, u64 ndim, u64 num_elements) {       \
    const float extreme_val = c_data[0];                                                           \
    float grad = dc_data[0];                                                                       \
    u64 count = 0;                                                                                 \
    u64 coords[MAX_NDIM];                                                                          \
    for (u64 i = 0; i < num_elements; ++i) {                                                       \
      linear_to_coords(i, shape, ndim, coords);                                                    \
      u64 a_offset = get_offset(coords, a_strides, ndim);                                          \
      if (a_data[a_offset] == extreme_val)                                                         \
        count++;                                                                                   \
    }                                                                                              \
    if (count == 0)                                                                                \
      return;                                                                                      \
    grad /= count;                                                                                 \
    _Pragma("omp parallel for private(coords)") for (u64 i = 0; i < num_elements; ++i) {           \
      linear_to_coords(i, shape, ndim, coords);                                                    \
      u64 a_offset = get_offset(coords, a_strides, ndim);                                          \
      u64 da_offset = get_offset(coords, da_strides, ndim);                                        \
      if (a_data[a_offset] == extreme_val)                                                         \
        da_data[da_offset] += grad;                                                                \
    }                                                                                              \
  }                                                                                                \
  void op_name##_cpu_backward_float_dim_kernel(                                                    \
      const float *a_data, const u64 *a_strides, const u64 *a_shape, u64 a_ndim,                   \
      const float *c_data, const u64 *c_strides, const u64 *c_shape, u64 c_ndim,                   \
      const float *dc_data, float *da_data, const u64 *da_strides, u64 dim, bool keepdim) {        \
    u64 output_num_elements = 1;                                                                   \
    for (u64 i = 0; i < c_ndim; ++i)                                                               \
      output_num_elements *= c_shape[i];                                                           \
    _Pragma("omp parallel for") for (u64 i = 0; i < output_num_elements; ++i) {                    \
      u64 c_coords[MAX_NDIM];                                                                      \
      linear_to_coords(i, c_shape, c_ndim, c_coords);                                              \
      const float extreme_val = c_data[get_offset(c_coords, c_strides, c_ndim)];                   \
      float grad = dc_data[get_offset(c_coords, c_strides, c_ndim)];                               \
      u64 count = 0;                                                                               \
      u64 a_coords[MAX_NDIM];                                                                      \
      reduction_output_to_input_coords(c_coords, a_coords, a_ndim, dim, keepdim);                  \
      u64 dim_len = a_shape[dim];                                                                  \
      for (u64 k = 0; k < dim_len; ++k) {                                                          \
        a_coords[dim] = k;                                                                         \
        u64 a_offset = get_offset(a_coords, a_strides, a_ndim);                                    \
        if (a_data[a_offset] == extreme_val)                                                       \
          count++;                                                                                 \
      }                                                                                            \
      if (count == 0)                                                                              \
        continue;                                                                                  \
      grad /= count;                                                                               \
      for (u64 k = 0; k < dim_len; ++k) {                                                          \
        a_coords[dim] = k;                                                                         \
        u64 a_offset = get_offset(a_coords, a_strides, a_ndim);                                    \
        u64 da_offset = get_offset(a_coords, da_strides, a_ndim);                                  \
        if (a_data[a_offset] == extreme_val)                                                       \
          da_data[da_offset] += grad;                                                              \
      }                                                                                            \
    }                                                                                              \
  }                                                                                                \
  void op_name##_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params) {        \
    const Tensor *a = inputs[0];                                                                   \
    u64 dim = params.dim;                                                                          \
    bool keepdim = params.keepdim;                                                                 \
    if (a->requires_grad) {                                                                        \
      if (dim == MAX_NDIM + 1) {                                                                   \
        u64 num_elements = numel(a);                                                               \
        if (is_contiguous(a)) {                                                                    \
          DISPATCH_DTYPE(a->dtype, op_name##_cpu_backward, _contig_kernel,                         \
                         ((const float *)output->grad->data, (const float *)a->data,               \
                          (float *)a->grad->data, num_elements));                                  \
        } else {                                                                                   \
          DISPATCH_DTYPE(a->dtype, op_name##_cpu_backward, _non_contig_kernel,                     \
                         ((const float *)a->data, a->strides, (const float *)output->data,         \
                          (const float *)output->grad->data, (float *)a->grad->data,               \
                          a->grad->strides, a->shape, a->ndim, num_elements));                     \
        }                                                                                          \
      } else {                                                                                     \
        DISPATCH_DTYPE(a->dtype, op_name##_cpu_backward, _dim_kernel,                              \
                       ((const float *)a->data, a->strides, a->shape, a->ndim,                     \
                        (const float *)output->data, output->strides, output->shape, output->ndim, \
                        (const float *)output->grad->data, (float *)a->grad->data,                 \
                        a->grad->strides, dim, keepdim));                                          \
      }                                                                                            \
    }                                                                                              \
  }

// ── SUM ──
DEFINE_REDUCE_CPU_FORWARD(sum, 0.0f, REDUCE_ADD, POST_SUM)
DEFINE_REDUCE_CPU_BACKWARD_SUM_MEAN(sum, SCALE_SUM)

// ── MEAN ──
DEFINE_REDUCE_CPU_FORWARD(mean, 0.0f, REDUCE_ADD, POST_MEAN)
DEFINE_REDUCE_CPU_BACKWARD_SUM_MEAN(mean, SCALE_MEAN)

// ── MAX ──
DEFINE_REDUCE_CPU_FORWARD(max, -FLT_MAX, REDUCE_MAX, POST_MAX)
DEFINE_REDUCE_CPU_BACKWARD_MAX_MIN(max, -FLT_MAX, COMPARE_MAX)

// ── MIN ──
DEFINE_REDUCE_CPU_FORWARD(min, FLT_MAX, REDUCE_MIN, POST_MIN)
DEFINE_REDUCE_CPU_BACKWARD_MAX_MIN(min, FLT_MAX, COMPARE_MIN)
