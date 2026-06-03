#include "kernels/leaky_relu.h"
#include "kernels/cpu_utils.h"
#include "tensor.h"
#include <math.h>
#include <omp.h>
#include <stdarg.h>

void leaky_relu_cpu_forward_float_contig_kernel(const float *a, float *c, u64 num_elements,
                                                float alpha) {
  u64 i = 0;
  __m256 alpha_vec = _mm256_set1_ps(alpha);
  __m256 zero_vec = _mm256_set1_ps(0.0f);

  for (; i + SIMD_WIDTH - 1 < num_elements; i += SIMD_WIDTH) {
    __m256 x = _mm256_loadu_ps(a + i);
    __m256 mask = _mm256_cmp_ps(x, zero_vec, _CMP_GT_OQ); // x > 0
    __m256 res = _mm256_blendv_ps(_mm256_mul_ps(x, alpha_vec), x,
                                  mask); // if mask is true (x>0) use x, else x*alpha
    _mm256_storeu_ps(c + i, res);
  }

  for (; i < num_elements; ++i) {
    c[i] = a[i] > 0 ? a[i] : a[i] * alpha;
  }
}

void leaky_relu_cpu_forward_float_non_contig_kernel(const float *a_data, const u64 *a_strides,
                                                    float *c_data, const u64 *c_strides,
                                                    const u64 *shape, u64 ndim, u64 num_elements,
                                                    float alpha) {
  u64 coords[MAX_NDIM];
#pragma omp parallel for private(coords)
  for (u64 i = 0; i < num_elements; ++i) {
    linear_to_coords(i, shape, ndim, coords);
    u64 a_offset = get_offset(coords, a_strides, ndim);
    u64 c_offset = get_offset(coords, c_strides, ndim);
    c_data[c_offset] = a_data[a_offset] > 0 ? a_data[a_offset] : a_data[a_offset] * alpha;
  }
}

void leaky_relu_cpu_backward_float_contig_kernel(const float *dout, const float *a, float *da,
                                                 u64 num_elements, float alpha) {
  u64 i = 0;
  __m256 alpha_vec = _mm256_set1_ps(alpha);
  __m256 one_vec = _mm256_set1_ps(1.0f);
  __m256 zero_vec = _mm256_set1_ps(0.0f);

  for (; i + SIMD_WIDTH - 1 < num_elements; i += SIMD_WIDTH) {
    __m256 x = _mm256_loadu_ps(a + i);
    __m256 out_grad = _mm256_loadu_ps(dout + i);
    __m256 mask = _mm256_cmp_ps(x, zero_vec, _CMP_GT_OQ); // x > 0
    __m256 grad_multiplier =
        _mm256_blendv_ps(alpha_vec, one_vec, mask); // if mask is true (x>0) use 1, else alpha

    if (da) {
      __m256 a_grad = _mm256_loadu_ps(da + i);
      __m256 new_grad = _mm256_add_ps(a_grad, _mm256_mul_ps(out_grad, grad_multiplier));
      _mm256_storeu_ps(da + i, new_grad);
    }
  }

  for (; i < num_elements; ++i) {
    if (da) {
      float grad_multiplier = a[i] > 0 ? 1.0f : alpha;
      da[i] += dout[i] * grad_multiplier;
    }
  }
}

void leaky_relu_cpu_backward_float_non_contig_kernel(const float *dout_data,
                                                     const u64 *dout_strides, const float *a_data,
                                                     const u64 *a_strides, float *da_data,
                                                     const u64 *da_strides, const u64 *shape,
                                                     u64 ndim, u64 num_elements, float alpha) {
  u64 coords[MAX_NDIM];
#pragma omp parallel for private(coords)
  for (u64 i = 0; i < num_elements; ++i) {
    linear_to_coords(i, shape, ndim, coords);
    u64 dout_offset = get_offset(coords, dout_strides, ndim);
    u64 a_offset = get_offset(coords, a_strides, ndim);
    u64 da_offset = get_offset(coords, da_strides, ndim);

    if (da_data) {
      float grad_multiplier = a_data[a_offset] > 0 ? 1.0f : alpha;
      da_data[da_offset] += dout_data[dout_offset] * grad_multiplier;
    }
  }
}

void leaky_relu_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params) {
  const Tensor *a = inputs[0];
  float alpha = params.fval;
  int num_elements = numel(a);

  if (is_contiguous(a) && is_contiguous(output)) {
    switch (a->dtype) {
    case FLOAT32:
      leaky_relu_cpu_forward_float_contig_kernel((const float *)a->data, (float *)output->data,
                                                 num_elements, alpha);
      break;
    default:
      break;
    }
  } else {
    switch (a->dtype) {
    case FLOAT32:
      leaky_relu_cpu_forward_float_non_contig_kernel((const float *)a->data, a->strides,
                                                     (float *)output->data, output->strides,
                                                     a->shape, a->ndim, num_elements, alpha);
      break;
    default:
      break;
    }
  }
}

void leaky_relu_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params) {
  const Tensor *a = inputs[0];
  float alpha = params.fval;
  int num_elements = numel(a);

  if (is_contiguous(a) && is_contiguous(output)) {
    switch (a->dtype) {
    case FLOAT32:
      leaky_relu_cpu_backward_float_contig_kernel(
          (const float *)output->grad->data, (const float *)a->data,
          a->requires_grad ? (float *)a->grad->data : NULL, num_elements, alpha);
      break;
    default:
      break;
    }
  } else {
    switch (a->dtype) {
    case FLOAT32:
      leaky_relu_cpu_backward_float_non_contig_kernel(
          (const float *)output->grad->data, output->grad->strides, (const float *)a->data,
          a->strides, a->requires_grad ? (float *)a->grad->data : NULL,
          a->requires_grad ? a->grad->strides : NULL, a->shape, a->ndim, num_elements, alpha);
      break;
    default:
      break;
    }
  }
}
