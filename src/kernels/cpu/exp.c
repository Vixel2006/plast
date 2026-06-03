#include "kernels/exp.h"
#include "kernels/cpu_utils.h"
#include "tensor.h"
#include <math.h>
#include <omp.h>

void exp_cpu_forward_float_contig_kernel(const float *a, float *c, u64 num_elements) {
  for (u64 i = 0; i < num_elements; ++i) {
    c[i] = expf(a[i]);
  }
}

void exp_cpu_forward_float_non_contig_kernel(const float *a_data, const u64 *a_strides,
                                             float *c_data, const u64 *c_strides, const u64 *shape,
                                             u64 ndim, u64 num_elements) {
  u64 coords[MAX_NDIM];
#pragma omp parallel for private(coords)
  for (u64 i = 0; i < num_elements; ++i) {
    linear_to_coords(i, shape, ndim, coords);
    u64 a_offset = get_offset(coords, a_strides, ndim);
    u64 c_offset = get_offset(coords, c_strides, ndim);
    c_data[c_offset] = expf(a_data[a_offset]);
  }
}

void exp_cpu_backward_float_contig_kernel(const float *dout, const float *a, float *da,
                                          u64 num_elements) {
  u64 i = 0;
  for (; i < num_elements; ++i) {
    if (da)
      da[i] += dout[i] * expf(a[i]);
  }
}

void exp_cpu_backward_float_non_contig_kernel(const float *dout_data, const u64 *dout_strides,
                                              const float *a_data, const u64 *a_strides,
                                              float *da_data, const u64 *da_strides,
                                              const u64 *shape, u64 ndim, u64 num_elements) {
  u64 coords[MAX_NDIM];
#pragma omp parallel for private(coords)
  for (u64 i = 0; i < num_elements; ++i) {
    linear_to_coords(i, shape, ndim, coords);
    u64 dout_offset = get_offset(coords, dout_strides, ndim);
    u64 a_offset = get_offset(coords, a_strides, ndim);
    u64 da_offset = get_offset(coords, da_strides, ndim);

    if (da_data)
      da_data[da_offset] += dout_data[dout_offset] * expf(a_data[a_offset]);
  }
}

void exp_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params) {
  const Tensor *a = inputs[0];
  int num_elements = numel(a);

  if (is_contiguous(a) && is_contiguous(output)) {
    switch (a->dtype) {
    case FLOAT32:
      exp_cpu_forward_float_contig_kernel((const float *)a->data, (float *)output->data,
                                          num_elements);
      break;
    default:
      break;
    }
  } else {
    switch (a->dtype) {
    case FLOAT32:
      exp_cpu_forward_float_non_contig_kernel((const float *)a->data, a->strides,
                                              (float *)output->data, output->strides, a->shape,
                                              a->ndim, num_elements);
      break;
    default:
      break;
    }
  }
}

void exp_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params) {
  const Tensor *a = inputs[0];
  int num_elements = numel(a);

  if (is_contiguous(a) && is_contiguous(output)) {
    switch (a->dtype) {
    case FLOAT32:
      exp_cpu_backward_float_contig_kernel(
          (const float *)output->grad->data, (const float *)a->data,
          a->requires_grad ? (float *)a->grad->data : NULL, num_elements);
      break;
    default:
      break;
    }
  } else {
    switch (a->dtype) {
    case FLOAT32:
      exp_cpu_backward_float_non_contig_kernel(
          (const float *)output->grad->data, output->grad->strides, (const float *)a->data,
          a->strides, a->requires_grad ? (float *)a->grad->data : NULL,
          a->requires_grad ? a->grad->strides : NULL, a->shape, a->ndim, num_elements);
      break;
    default:
      break;
    }
  }
}
