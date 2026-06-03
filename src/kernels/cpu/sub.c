#include "kernels/sub.h"
#include "kernels/cpu_utils.h"
#include "tensor.h"
#include <omp.h>

void sub_cpu_forward_float_contig_kernel(const float *a, const float *b,
                                         float *c, u64 num_elements) {
  u64 i = 0;
  for (; i + SIMD_WIDTH - 1 < num_elements; i += SIMD_WIDTH) {
    __m256 x = _mm256_loadu_ps(a + i);
    __m256 y = _mm256_loadu_ps(b + i);
    __m256 z = _mm256_sub_ps(x, y);
    _mm256_storeu_ps(c + i, z);
  }

  for (; i < num_elements; ++i) {
    c[i] = a[i] - b[i];
  }
}

void sub_cpu_forward_float_non_contig_kernel(
    const float *a_data, const u64 *a_strides, const float *b_data,
    const u64 *b_strides, float *c_data, const u64 *c_strides, const u64 *shape,
    u64 ndim, u64 num_elements) {
  u64 coords[MAX_NDIM];
#pragma omp parallel for private(coords)
  for (u64 i = 0; i < num_elements; ++i) {
    linear_to_coords(i, shape, ndim, coords);
    u64 a_offset = get_offset(coords, a_strides, ndim);
    u64 b_offset = get_offset(coords, b_strides, ndim);
    u64 c_offset = get_offset(coords, c_strides, ndim);
    c_data[c_offset] = a_data[a_offset] - b_data[b_offset];
  }
}

void sub_cpu_backward_float_contig_kernel(const float *dout, float *da,
                                          float *db, u64 num_elements) {
  u64 i = 0;
  for (; i + SIMD_WIDTH - 1 < num_elements; i += SIMD_WIDTH) {
    __m256 out_grad = _mm256_loadu_ps(dout + i);
    if (da) {
      __m256 a_grad = _mm256_loadu_ps(da + i);
      __m256 new_grad = _mm256_add_ps(out_grad, a_grad);
      _mm256_storeu_ps(da + i, new_grad);
    }
    if (db) {
      __m256 b_grad = _mm256_loadu_ps(db + i);
      __m256 new_grad = _mm256_sub_ps(b_grad, out_grad); // db -= dout
      _mm256_storeu_ps(db + i, new_grad);
    }
  }

  for (; i < num_elements; ++i) {
    if (da)
      da[i] += dout[i];
    if (db)
      db[i] -= dout[i];
  }
}

void sub_cpu_backward_float_non_contig_kernel(
    const float *dout_data, const u64 *dout_strides, float *da_data,
    const u64 *da_strides, float *db_data, const u64 *db_strides,
    const u64 *shape, u64 ndim, u64 num_elements) {
  u64 coords[MAX_NDIM];
#pragma omp parallel for private(coords)
  for (u64 i = 0; i < num_elements; ++i) {
    linear_to_coords(i, shape, ndim, coords);
    u64 dout_offset = get_offset(coords, dout_strides, ndim);
    u64 da_offset = get_offset(coords, da_strides, ndim);
    u64 db_offset = get_offset(coords, db_strides, ndim);

    if (da_data)
      da_data[da_offset] += dout_data[dout_offset];
    if (db_data)
      db_data[db_offset] -= dout_data[dout_offset];
  }
}

void sub_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params) {
  const Tensor *a = inputs[0];
  const Tensor *b = inputs[1];

  int num_elements = numel(a);

  if (is_contiguous(a) && is_contiguous(b) && is_contiguous(output)) {
    switch (a->dtype) {
    case FLOAT32:
      sub_cpu_forward_float_contig_kernel((const float *)a->data,
                                          (const float *)b->data,
                                          (float *)output->data, num_elements);
      break;
    default:
      break;
    }
  } else {
    switch (a->dtype) {
    case FLOAT32:
      sub_cpu_forward_float_non_contig_kernel(
          (const float *)a->data, a->strides, (const float *)b->data,
          b->strides, (float *)output->data, output->strides, a->shape, a->ndim,
          num_elements);
      break;
    default:
      break;
    }
  }
}

void sub_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params) {
  const Tensor *a = inputs[0];
  const Tensor *b = inputs[1];

  int num_elements = numel(a);

  if (is_contiguous(a) && is_contiguous(b) && is_contiguous(output)) {
    switch (a->dtype) {
    case FLOAT32:
      sub_cpu_backward_float_contig_kernel(
          (const float *)output->grad->data,
          a->requires_grad ? (float *)a->grad->data : NULL,
          b->requires_grad ? (float *)b->grad->data : NULL, num_elements);
      break;
    default:
      break;
    }
  } else {
    switch (a->dtype) {
    case FLOAT32:
      sub_cpu_backward_float_non_contig_kernel(
          (const float *)output->grad->data, output->grad->strides,
          a->requires_grad ? (float *)a->grad->data : NULL,
          a->requires_grad ? a->grad->strides : NULL,
          b->requires_grad ? (float *)b->grad->data : NULL,
          b->requires_grad ? b->grad->strides : NULL, a->shape, a->ndim,
          num_elements);
      break;
    default:
      break;
    }
  }
}
