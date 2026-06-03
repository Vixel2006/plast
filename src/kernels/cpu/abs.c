#include "kernels/abs.h"
#include "kernels/cpu_utils.h"
#include "tensor.h"
#include <math.h>
#include <omp.h>

void abs_cpu_forward_float_contig_kernel(const float *a, float *c,
                                         u64 num_elements) {
  u64 i = 0;
  for (; i + SIMD_WIDTH - 1 < num_elements; i += SIMD_WIDTH) {

    __m256 x = _mm256_loadu_ps(a + i);

    // NOTE: Abs can be done by applying a mask that has 0 in the sign bit, then
    // by applying the logical and on the number we get the absolute value
    // http://steve.hollasch.net/cgindex/coding/ieeefloat.html for understanding
    // floating-point layout

    __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 y = _mm256_and_ps(x, abs_mask);
    _mm256_storeu_ps(c + i, y);
  }

  for (; i < num_elements; ++i) {
    c[i] = fabsf(a[i]);
  }
}

void abs_cpu_forward_float_non_contig_kernel(
    const float *a_data, const u64 *a_strides, float *c_data,
    const u64 *c_strides, const u64 *shape, u64 ndim, u64 num_elements) {
  u64 coords[MAX_NDIM];
#pragma omp parallel for private(coords)
  for (u64 i = 0; i < num_elements; ++i) {
    linear_to_coords(i, shape, ndim, coords);
    u64 a_offset = get_offset(coords, a_strides, ndim);
    u64 c_offset = get_offset(coords, c_strides, ndim);
    c_data[c_offset] = fabsf(a_data[a_offset]);
  }
}

void abs_cpu_backward_float_contig_kernel(const float *dout, const float *a,
                                          float *da, u64 num_elements) {
  u64 i = 0;
  __m256 zero_vec = _mm256_set1_ps(0.0f);
  __m256 one_vec = _mm256_set1_ps(1.0f);
  __m256 neg_one_vec = _mm256_set1_ps(-1.0f);

  for (; i + SIMD_WIDTH - 1 < num_elements; i += SIMD_WIDTH) {
    __m256 x = _mm256_loadu_ps(a + i);
    __m256 out_grad = _mm256_loadu_ps(dout + i);

    __m256 mask_gt_zero = _mm256_cmp_ps(x, zero_vec, _CMP_GT_OQ); // x > 0
    __m256 mask_lt_zero = _mm256_cmp_ps(x, zero_vec, _CMP_LT_OQ); // x < 0

    __m256 grad_multiplier = _mm256_set1_ps(0.0f);
    grad_multiplier = _mm256_blendv_ps(
        grad_multiplier, one_vec, mask_gt_zero); // if x > 0, multiplier is 1
    grad_multiplier =
        _mm256_blendv_ps(grad_multiplier, neg_one_vec,
                         mask_lt_zero); // if x < 0, multiplier is -1

    if (da) {
      __m256 a_grad = _mm256_loadu_ps(da + i);
      __m256 new_grad =
          _mm256_add_ps(a_grad, _mm256_mul_ps(out_grad, grad_multiplier));
      _mm256_storeu_ps(da + i, new_grad);
    }
  }

  for (; i < num_elements; ++i) {
    if (da) {
      float grad_multiplier = (a[i] > 0) ? 1.0f : ((a[i] < 0) ? -1.0f : 0.0f);
      da[i] += dout[i] * grad_multiplier;
    }
  }
}

void abs_cpu_backward_float_non_contig_kernel(
    const float *dout_data, const u64 *dout_strides, const float *a_data,
    const u64 *a_strides, float *da_data, const u64 *da_strides,
    const u64 *shape, u64 ndim, u64 num_elements) {
  u64 coords[MAX_NDIM];
#pragma omp parallel for private(coords)
  for (u64 i = 0; i < num_elements; ++i) {
    linear_to_coords(i, shape, ndim, coords);
    u64 dout_offset = get_offset(coords, dout_strides, ndim);
    u64 a_offset = get_offset(coords, a_strides, ndim);
    u64 da_offset = get_offset(coords, da_strides, ndim);

    if (da_data) {
      float grad_multiplier = (a_data[a_offset] > 0)
                                  ? 1.0f
                                  : ((a_data[a_offset] < 0) ? -1.0f : 0.0f);
      da_data[da_offset] += dout_data[dout_offset] * grad_multiplier;
    }
  }
}

void abs_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params) {
  const Tensor *a = inputs[0];
  int num_elements = numel(a);

  if (is_contiguous(a) && is_contiguous(output)) {
    switch (a->dtype) {
    case FLOAT32:
      abs_cpu_forward_float_contig_kernel((const float *)a->data,
                                          (float *)output->data, num_elements);
      break;
    default:
      break;
    }
  } else {
    switch (a->dtype) {
    case FLOAT32:
      abs_cpu_forward_float_non_contig_kernel(
          (const float *)a->data, a->strides, (float *)output->data,
          output->strides, a->shape, a->ndim, num_elements);
      break;
    default:
      break;
    }
  }
}

void abs_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params) {
  const Tensor *a = inputs[0];
  int num_elements = numel(a);

  if (is_contiguous(a) && is_contiguous(output)) {
    switch (a->dtype) {
    case FLOAT32:
      abs_cpu_backward_float_contig_kernel(
          (const float *)output->grad->data, (const float *)a->data,
          a->requires_grad ? (float *)a->grad->data : NULL, num_elements);
      break;
    default:
      break;
    }
  } else {
    switch (a->dtype) {
    case FLOAT32:
      abs_cpu_backward_float_non_contig_kernel(
          (const float *)output->grad->data, output->grad->strides,
          (const float *)a->data, a->strides,
          a->requires_grad ? (float *)a->grad->data : NULL,
          a->requires_grad ? a->grad->strides : NULL, a->shape, a->ndim,
          num_elements);
      break;
    default:
      break;
    }
  }
}
