#include "kernels/max.h"
#include "kernels/cpu_utils.h"
#include <float.h>
#include <omp.h>
#include <stdarg.h>
#include <string.h>

void max_cpu_forward_float_contig_kernel(const float *a, float *c,
                                         u64 num_elements) {
  float max_val = -FLT_MAX;
#pragma omp parallel for reduction(max : max_val)
  for (u64 i = 0; i < num_elements; ++i) {
    if (a[i] > max_val)
      max_val = a[i];
  }
  c[0] = max_val;
}

void max_cpu_backward_float_contig_kernel(const float *a, const float *c,
                                          const float *dc, float *da,
                                          u64 num_elements) {
  const float max = c[0];
  float grad = dc[0];
  u64 count = 0;

#pragma omp parallel for reduction(+ : count)
  for (u64 i = 0; i < num_elements; ++i) {
    if (a[i] == max)
      ++count;
  }

  if (count == 0)
    return;

  grad /= count;

#pragma omp parallel for
  for (u64 i = 0; i < num_elements; ++i) {
    if (a[i] == max)
      da[i] += grad;
  }
}

void max_cpu_forward_float_non_contig_kernel(const float *a_data,
                                             const u64 *a_strides,
                                             const u64 *shape, u64 ndim,
                                             u64 num_elements, float *c_data) {
  float max_val = -FLT_MAX;
  u64 coords[MAX_NDIM];

#pragma omp parallel for private(coords) reduction(max : max_val)
  for (u64 i = 0; i < num_elements; ++i) {
    linear_to_coords(i, shape, ndim, coords);
    u64 a_offset = get_offset(coords, a_strides, ndim);
    if (a_data[a_offset] > max_val) {
      max_val = a_data[a_offset];
    }
  }
  c_data[0] = max_val;
}

void max_cpu_backward_float_non_contig_kernel(
    const float *a_data, const u64 *a_strides, const float *c_data,
    const float *dc_data, float *da_data, const u64 *da_strides,
    const u64 *shape, u64 ndim, u64 num_elements) {
  const float max = c_data[0];
  float grad = dc_data[0];
  u64 count = 0;
  u64 coords[MAX_NDIM];

#pragma omp parallel for private(coords) reduction(+ : count)
  for (u64 i = 0; i < num_elements; ++i) {
    linear_to_coords(i, shape, ndim, coords);
    u64 a_offset = get_offset(coords, a_strides, ndim);
    if (a_data[a_offset] == max)
      ++count;
  }

  if (count == 0)
    return;

  grad /= count;

#pragma omp parallel for private(coords)
  for (u64 i = 0; i < num_elements; ++i) {
    linear_to_coords(i, shape, ndim, coords);
    u64 a_offset = get_offset(coords, a_strides, ndim);
    u64 da_offset = get_offset(coords, da_strides, ndim);
    if (a_data[a_offset] == max)
      da_data[da_offset] += grad;
  }
}

void max_cpu_forward_float_dim_kernel(const float *a_data, const u64 *a_strides,
                                        const u64 *a_shape, u64 a_ndim,
                                        float *c_data, const u64 *c_strides,
                                        const u64 *c_shape, u64 c_ndim,
                                        u64 dim, bool keepdim) {
  u64 output_num_elements = 1;
  for (u64 i = 0; i < c_ndim; ++i) {
    output_num_elements *= c_shape[i];
  }

#pragma omp parallel for
  for (u64 i = 0; i < output_num_elements; ++i) {
    u64 c_coords[MAX_NDIM];
    linear_to_coords(i, c_shape, c_ndim, c_coords);

    float max_val = -FLT_MAX;

    u64 a_coords[MAX_NDIM];
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

    u64 dim_len = a_shape[dim];
    for (u64 k = 0; k < dim_len; ++k) {
      a_coords[dim] = k;
      u64 a_offset = get_offset(a_coords, a_strides, a_ndim);
      if (a_data[a_offset] > max_val) {
        max_val = a_data[a_offset];
      }
    }
    c_data[i] = max_val;
  }
}

void max_cpu_backward_float_dim_kernel(
    const float *a_data, const u64 *a_strides, const u64 *a_shape, u64 a_ndim,
    const float *c_data, const u64 *c_strides, const u64 *c_shape, u64 c_ndim,
    const float *dc_data, float *da_data, const u64 *da_strides, u64 dim, bool keepdim) {
  u64 output_num_elements = 1;
  for (u64 i = 0; i < c_ndim; ++i) {
    output_num_elements *= c_shape[i];
  }

#pragma omp parallel for
  for (u64 i = 0; i < output_num_elements; ++i) {
    u64 c_coords[MAX_NDIM];
    linear_to_coords(i, c_shape, c_ndim, c_coords);

    const float max_val = c_data[get_offset(c_coords, c_strides, c_ndim)];
    float grad = dc_data[get_offset(c_coords, c_strides, c_ndim)];

    u64 count = 0;
    u64 a_coords[MAX_NDIM];
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

    u64 dim_len = a_shape[dim];
    for (u64 k = 0; k < dim_len; ++k) {
      a_coords[dim] = k;
      u64 a_offset = get_offset(a_coords, a_strides, a_ndim);
      if (a_data[a_offset] == max_val) {
        count++;
      }
    }

    if (count == 0)
      continue;

    grad /= count;

    for (u64 k = 0; k < dim_len; ++k) {
      a_coords[dim] = k;
      u64 a_offset = get_offset(a_coords, a_strides, a_ndim);
      u64 da_offset = get_offset(a_coords, da_strides, a_ndim);
      if (a_data[a_offset] == max_val) {
        da_data[da_offset] += grad;
      }
    }
  }
}

void max_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params) {
  const Tensor *a = inputs[0];
  u64 dim = params.dim;
  bool keepdim = params.keepdim;

  if (dim == MAX_NDIM + 1) {
    u64 num_elements = numel(a);

    if (is_contiguous(a)) {
      switch (a->dtype) {
      case FLOAT32:
        max_cpu_forward_float_contig_kernel((const float *)a->data,
                                            (float *)output->data,
                                            num_elements);
        break;
      default:
        break;
      }
    } else {
      switch (a->dtype) {
      case FLOAT32:
        max_cpu_forward_float_non_contig_kernel(
            (const float *)a->data, a->strides, a->shape, a->ndim,
            num_elements, (float *)output->data);
        break;
      default:
        break;
      }
    }
  } else {
    compute_reduction_shape_strides(a->shape, a->ndim, dim, keepdim,
                                    output->shape, &output->ndim, output->strides);

    switch (a->dtype) {
    case FLOAT32:
      max_cpu_forward_float_dim_kernel((const float *)a->data, a->strides,
                                       a->shape, a->ndim, (float *)output->data,
                                       output->strides, output->shape,
                                       output->ndim, dim, keepdim);
      break;
    default:
      break;
    }
  }
}

void max_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params) {
  const Tensor *a = inputs[0];
  u64 dim = params.dim;
  bool keepdim = params.keepdim;

  if (a->requires_grad) {
    if (dim == MAX_NDIM + 1) {
      u64 num_elements = numel(a);

      if (is_contiguous(a)) {
        switch (a->dtype) {
        case FLOAT32:
          max_cpu_backward_float_contig_kernel(
              (const float *)a->data, (const float *)output->data,
              (const float *)output->grad->data, (float *)a->grad->data,
              num_elements);
          break;
        default:
          break;
        }
      } else {
        switch (a->dtype) {
        case FLOAT32:
          max_cpu_backward_float_non_contig_kernel(
              (const float *)a->data, a->strides, (const float *)output->data,
              (const float *)output->grad->data, (float *)a->grad->data,
              a->grad->strides, a->shape, a->ndim, num_elements);
          break;
        default:
          break;
        }
      }
    } else {
      switch (a->dtype) {
      case FLOAT32:
        max_cpu_backward_float_dim_kernel(
            (const float *)a->data, a->strides, a->shape, a->ndim,
            (const float *)output->data, output->strides, output->shape,
            output->ndim, (const float *)output->grad->data,
            (float *)a->grad->data, a->grad->strides, dim, keepdim);
        break;
      default:
        break;
      }
    }
  }
}