#include "kernels/cuda/cuda_utils.cuh"
#include "kernels/sub.h"
#include <cuda_runtime.h>
#include <stdarg.h>

__global__ void sub_cuda_forward_float_contig_kernel(const float *a,
                                                     const float *b, float *c,
                                                     u64 num_elements) {
  u64 idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    c[idx] = a[idx] - b[idx];
  }
}

__global__ void sub_cuda_backward_float_contig_kernel(const float *dout,
                                                      float *da, float *db,
                                                      u64 num_elements) {
  u64 idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    if (da)
      da[idx] += dout[idx];
    if (db)
      db[idx] -= dout[idx];
  }
}

__global__ void sub_cuda_forward_float_non_contig_kernel(
    const float *a_data, const u64 *a_strides, const u64 *a_shape, u64 a_ndim,
    const float *b_data, const u64 *b_strides, const u64 *b_shape, u64 b_ndim,
    float *c_data, const u64 *c_strides, const u64 *c_shape, u64 c_ndim,
    u64 num_elements) {
  u64 linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear_idx < num_elements) {
    u64 coords[MAX_NDIM];
    cuda_linear_to_coords(linear_idx, c_shape, c_ndim, coords);
    u64 a_offset = cuda_get_offset_broadcast(coords, c_ndim, a_strides, a_shape, a_ndim);
    u64 b_offset = cuda_get_offset_broadcast(coords, c_ndim, b_strides, b_shape, b_ndim);
    u64 c_offset = cuda_get_offset(coords, c_strides, c_ndim);
    c_data[c_offset] = a_data[a_offset] - b_data[b_offset];
  }
}

__global__ void sub_cuda_backward_float_non_contig_kernel(
    const float *dout_data, const u64 *dout_strides, const u64 *dout_shape, u64 dout_ndim,
    float *da_data, const u64 *da_strides, const u64 *da_shape, u64 da_ndim,
    float *db_data, const u64 *db_strides, const u64 *db_shape, u64 db_ndim,
    u64 num_elements) {
  u64 linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear_idx < num_elements) {
    u64 coords[MAX_NDIM];
    cuda_linear_to_coords(linear_idx, dout_shape, dout_ndim, coords);
    u64 dout_offset = cuda_get_offset(coords, dout_strides, dout_ndim);
    float dout_val = dout_data[dout_offset];

    if (da_data) {
      u64 da_offset = cuda_get_offset_broadcast(coords, dout_ndim, da_strides, da_shape, da_ndim);
      atomicAdd(&da_data[da_offset], dout_val);
    }
    if (db_data) {
      u64 db_offset = cuda_get_offset_broadcast(coords, dout_ndim, db_strides, db_shape, db_ndim);
      atomicAdd(&db_data[db_offset], -dout_val);
    }
  }
}

void sub_cuda_forward(const Tensor **inputs, Tensor *output, ...) {
  const Tensor *a = inputs[0];
  const Tensor *b = inputs[1];
  u64 num_elements = numel(output);

  int block_size = 256;
  int grid_size = (num_elements + block_size - 1) / block_size;

  if (is_contiguous(a) && is_contiguous(b) && is_contiguous(output) &&
      a->ndim == b->ndim && a->ndim == output->ndim) {
    bool shapes_match = true;
    for (u64 i = 0; i < a->ndim; ++i) {
      if (a->shape[i] != b->shape[i] || a->shape[i] != output->shape[i]) {
        shapes_match = false;
        break;
      }
    }

    if (shapes_match) {
      switch (a->dtype) {
      case FLOAT32:
        sub_cuda_forward_float_contig_kernel<<<grid_size, block_size>>>(
            (const float *)a->data, (const float *)b->data, (float *)output->data,
            num_elements);
        break;
      default:
        break;
      }
      return;
    }
  }

  switch (a->dtype) {
  case FLOAT32:
    sub_cuda_forward_float_non_contig_kernel<<<grid_size, block_size>>>(
        (const float *)a->data, a->strides, a->shape, a->ndim,
        (const float *)b->data, b->strides, b->shape, b->ndim,
        (float *)output->data, output->strides, output->shape, output->ndim,
        num_elements);
    break;
  default:
    break;
  }
  cudaDeviceSynchronize();
}

void sub_cuda_backward(Tensor **inputs, const Tensor *output, ...) {
  const Tensor *a = inputs[0];
  const Tensor *b = inputs[1];
  u64 num_elements = numel(output);

  int block_size = 256;
  int grid_size = (num_elements + block_size - 1) / block_size;

  switch (a->dtype) {
  case FLOAT32:
    sub_cuda_backward_float_non_contig_kernel<<<grid_size, block_size>>>(
        (const float *)output->grad->data, output->grad->strides, output->grad->shape, output->grad->ndim,
        a->requires_grad ? (float *)a->grad->data : NULL, a->grad ? a->grad->strides : NULL, a->grad ? a->grad->shape : NULL, a->grad ? a->grad->ndim : 0,
        b->requires_grad ? (float *)b->grad->data : NULL, b->grad ? b->grad->strides : NULL, b->grad ? b->grad->shape : NULL, b->grad ? b->grad->ndim : 0,
        num_elements);
    break;
  default:
    break;
  }
  cudaDeviceSynchronize();
}
