#include "kernels/div.h"
#include "kernels/cpu_utils.h"
#include "tensor.h"
#include <omp.h>

void div_cpu_forward_float_kernel(
    const float *a_data, const u64 *a_strides, const u64 *a_shape, u64 a_ndim,
    const float *b_data, const u64 *b_strides, const u64 *b_shape, u64 b_ndim,
    float *c_data, const u64 *c_strides, const u64 *c_shape, u64 c_ndim)
{
    u64 num_elements = 1;
    for(u64 i=0; i<c_ndim; ++i) num_elements *= c_shape[i];

    u64 c_coords[MAX_NDIM];
    #pragma omp parallel for private(c_coords)
    for (u64 i = 0; i < num_elements; ++i) {
        linear_to_coords(i, c_shape, c_ndim, c_coords);

        // Calculate offset for a
        u64 a_offset = 0;
        int a_dim_offset = c_ndim - a_ndim;
        for (u64 j = 0; j < a_ndim; ++j) {
            u64 coord = c_coords[j + a_dim_offset];
            if (a_shape[j] == 1) {
                coord = 0;
            }
            a_offset += coord * a_strides[j];
        }

        // Calculate offset for b
        u64 b_offset = 0;
        int b_dim_offset = c_ndim - b_ndim;
        for (u64 j = 0; j < b_ndim; ++j) {
            u64 coord = c_coords[j + b_dim_offset];
            if (b_shape[j] == 1) {
                coord = 0;
            }
            b_offset += coord * b_strides[j];
        }

        u64 c_offset = get_offset(c_coords, c_strides, c_ndim);
        c_data[c_offset] = a_data[a_offset] / b_data[b_offset];
    }
}

void div_cpu_backward_float_kernel(
    const float *dout_data, const u64 *dout_strides, const u64 *dout_shape, u64 dout_ndim,
    const float *a_data, const u64 *a_strides, const u64 *a_shape, u64 a_ndim,
    const float *b_data, const u64 *b_strides, const u64 *b_shape, u64 b_ndim,
    float *da_data, const u64 *da_strides, const u64 *da_shape, u64 da_ndim,
    float *db_data, const u64 *db_strides, const u64 *db_shape, u64 db_ndim)
{
    u64 num_elements = 1;
    for(u64 i=0; i<dout_ndim; ++i) num_elements *= dout_shape[i];

    u64 dout_coords[MAX_NDIM];
    #pragma omp parallel for private(dout_coords)
    for (u64 i = 0; i < num_elements; ++i) {
        linear_to_coords(i, dout_shape, dout_ndim, dout_coords);
        u64 dout_offset = get_offset(dout_coords, dout_strides, dout_ndim);
        const float grad = dout_data[dout_offset];

        // Calculate offset for a
        u64 a_offset = 0;
        int a_dim_offset = dout_ndim - a_ndim;
        for (u64 j = 0; j < a_ndim; ++j) {
            u64 coord = dout_coords[j + a_dim_offset];
            if (a_shape[j] == 1) {
                coord = 0;
            }
            a_offset += coord * a_strides[j];
        }

        // Calculate offset for b
        u64 b_offset = 0;
        int b_dim_offset = dout_ndim - b_ndim;
        for (u64 j = 0; j < b_ndim; ++j) {
            u64 coord = dout_coords[j + b_dim_offset];
            if (b_shape[j] == 1) {
                coord = 0;
            }
            b_offset += coord * b_strides[j];
        }

        if (da_data) {
            u64 da_offset = 0;
            int da_dim_offset = dout_ndim - da_ndim;
            for (u64 j = 0; j < da_ndim; ++j) {
                u64 coord = dout_coords[j + da_dim_offset];
                if (da_shape[j] == 1) {
                    coord = 0;
                }
                da_offset += coord * da_strides[j];
            }
            #pragma omp atomic
            da_data[da_offset] += grad / b_data[b_offset];
        }

        if (db_data) {
            u64 db_offset = 0;
            int db_dim_offset = dout_ndim - db_ndim;
            for (u64 j = 0; j < db_ndim; ++j) {
                u64 coord = dout_coords[j + db_dim_offset];
                if (db_shape[j] == 1) {
                    coord = 0;
                }
                db_offset += coord * db_strides[j];
            }
            float b_val = b_data[b_offset];
            #pragma omp atomic
            db_data[db_offset] -= grad * a_data[a_offset] / (b_val * b_val);
        }
    }
}

void div_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params) {
  const Tensor *a = inputs[0];
  const Tensor *b = inputs[1];

  switch (a->dtype) {
    case FLOAT32:
      div_cpu_forward_float_kernel(
          (const float *)a->data, a->strides, a->shape, a->ndim,
          (const float *)b->data, b->strides, b->shape, b->ndim,
          (float *)output->data, output->strides, output->shape, output->ndim);
      break;
    default:
      break;
  }
}

void div_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params) {
    const Tensor *a = inputs[0];
    const Tensor *b = inputs[1];

    switch (a->dtype) {
    case FLOAT32:
      div_cpu_backward_float_kernel(
          (const float *)output->grad->data, output->grad->strides, output->grad->shape, output->grad->ndim,
          (const float *)a->data, a->strides, a->shape, a->ndim,
          (const float *)b->data, b->strides, b->shape, b->ndim,
          a->requires_grad ? (float *)a->grad->data : NULL, a->strides, a->shape, a->ndim,
          b->requires_grad ? (float *)b->grad->data : NULL, b->strides, b->shape, b->ndim);
      break;
    default:
      break;
    }
}