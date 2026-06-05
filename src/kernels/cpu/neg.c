#include "kernels/neg.h"
#include "kernels/cpu_utils.h"
#include "core/tensor.h"
#include <omp.h>

void neg_cpu_forward_float_contig_kernel(const float *a, float *c, u64 num_elements) {
  u64 i = 0;
  for (; i + SIMD_WIDTH - 1 < num_elements; i += SIMD_WIDTH) {
    __m256 x = _mm256_loadu_ps(a + i);
    __m256 z = _mm256_sub_ps(_mm256_set1_ps(0.0f), x); // 0 - x
    _mm256_storeu_ps(c + i, z);
  }

  for (; i < num_elements; ++i) {
    c[i] = -a[i];
  }
}

void neg_cpu_forward_float_non_contig_kernel(const float *a_data, const u64 *a_strides,
                                             float *c_data, const u64 *c_strides, const u64 *shape,
                                             u64 ndim, u64 num_elements) {
  u64 coords[MAX_NDIM];
#pragma omp parallel for private(coords)
  for (u64 i = 0; i < num_elements; ++i) {
    linear_to_coords(i, shape, ndim, coords);
    u64 a_offset = get_offset(coords, a_strides, ndim);
    u64 c_offset = get_offset(coords, c_strides, ndim);
    c_data[c_offset] = -a_data[a_offset];
  }
}

void neg_cpu_backward_float_contig_kernel(const float *dout, float *da, u64 num_elements) {
  u64 i = 0;
  for (; i + SIMD_WIDTH - 1 < num_elements; i += SIMD_WIDTH) {
    __m256 out_grad = _mm256_loadu_ps(dout + i);
    if (da) {
      __m256 a_grad = _mm256_loadu_ps(da + i);
      __m256 new_grad = _mm256_sub_ps(a_grad, out_grad); // da -= dout
      _mm256_storeu_ps(da + i, new_grad);
    }
  }

  for (; i < num_elements; ++i) {
    if (da)
      da[i] -= dout[i];
  }
}

void neg_cpu_backward_float_non_contig_kernel(const float *dout_data, const u64 *dout_strides,
                                              float *da_data, const u64 *da_strides,
                                              const u64 *shape, u64 ndim, u64 num_elements) {
  u64 coords[MAX_NDIM];
#pragma omp parallel for private(coords)
  for (u64 i = 0; i < num_elements; ++i) {
    linear_to_coords(i, shape, ndim, coords);
    u64 dout_offset = get_offset(coords, dout_strides, ndim);
    u64 da_offset = get_offset(coords, da_strides, ndim);

    if (da_data)
      da_data[da_offset] -= dout_data[dout_offset];
  }
}

void neg_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params) {
  const Tensor *a = inputs[0];
  int num_elements = numel(a);

  if (is_contiguous(a) && is_contiguous(output)) {
    switch (a->dtype) {
    case FLOAT32:
      neg_cpu_forward_float_contig_kernel((const float *)a->data, (float *)output->data,
                                          num_elements);
      break;
    default:
      break;
    }
  } else {
    switch (a->dtype) {
    case FLOAT32:
      neg_cpu_forward_float_non_contig_kernel((const float *)a->data, a->strides,
                                              (float *)output->data, output->strides, a->shape,
                                              a->ndim, num_elements);
      break;
    default:
      break;
    }
  }
}

void neg_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params) {
  const Tensor *a = inputs[0];
  int num_elements = numel(a);

  if (is_contiguous(a) && is_contiguous(output)) {
    switch (a->dtype) {
    case FLOAT32:
      neg_cpu_backward_float_contig_kernel((const float *)output->grad->data,
                                           a->requires_grad ? (float *)a->grad->data : NULL,
                                           num_elements);
      break;
    default:
      break;
    }
  } else {
    switch (a->dtype) {
    case FLOAT32:
      neg_cpu_backward_float_non_contig_kernel(
          (const float *)output->grad->data, output->grad->strides,
          a->requires_grad ? (float *)a->grad->data : NULL,
          a->requires_grad ? a->grad->strides : NULL, a->shape, a->ndim, num_elements);
      break;
    default:
      break;
    }
  }
}
