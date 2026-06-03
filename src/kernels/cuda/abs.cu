#include "kernels/abs.h"
#include "kernels/cuda/cuda_utils.cuh"
#include <cuda_runtime.h>
#include <math.h>
#include <stdarg.h>

__global__ void abs_cuda_forward_float_contig_kernel(const float *a, float *c,
                                                     u64 num_elements) {
  u64 idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    c[idx] = fabsf(a[idx]);
  }
}

__global__ void abs_cuda_backward_float_contig_kernel(const float *dout,
                                                      const float *a, float *da,
                                                      u64 num_elements) {
  u64 idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    if (da) {
      float grad_multiplier =
          (a[idx] > 0) ? 1.0f : ((a[idx] < 0) ? -1.0f : 0.0f);
      da[idx] += dout[idx] * grad_multiplier;
    }
  }
}

__global__ void abs_cuda_forward_float_non_contig_kernel(
    const float *a_data, const u64 *a_strides, float *c_data,
    const u64 *c_strides, const u64 *shape, u64 ndim, u64 num_elements) {
  u64 linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear_idx < num_elements) {
    u64 coords[MAX_NDIM];
    cuda_linear_to_coords(linear_idx, shape, ndim, coords);
    u64 a_offset = cuda_get_offset(coords, a_strides, ndim);
    u64 c_offset = cuda_get_offset(coords, c_strides, ndim);
    c_data[c_offset] = fabsf(a_data[a_offset]);
  }
}

__global__ void abs_cuda_backward_float_non_contig_kernel(
    const float *dout_data, const u64 *dout_strides, const float *a_data,
    const u64 *a_strides, float *da_data, const u64 *da_strides,
    const u64 *shape, u64 ndim, u64 num_elements) {
  u64 linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear_idx < num_elements) {
    u64 coords[MAX_NDIM];
    cuda_linear_to_coords(linear_idx, shape, ndim, coords);
    u64 dout_offset = cuda_get_offset(coords, dout_strides, ndim);
    u64 a_offset = cuda_get_offset(coords, a_strides, ndim);
    u64 da_offset = cuda_get_offset(coords, da_strides, ndim);

    if (da_data) {
      float grad_multiplier = (a_data[a_offset] > 0)
                                  ? 1.0f
                                  : ((a_data[a_offset] < 0) ? -1.0f : 0.0f);
      da_data[da_offset] += dout_data[dout_offset] * grad_multiplier;
    }
  }
}

void abs_cuda_forward(const Tensor **inputs, Tensor *output, KernelParams params) {
  const Tensor *a = inputs[0];
  u64 num_elements = numel(a);

  int block_size = 256;
  int grid_size = (num_elements + block_size - 1) / block_size;

  if (is_contiguous(a) && is_contiguous(output)) {
    switch (a->dtype) {
    case FLOAT32:
      abs_cuda_forward_float_contig_kernel<<<grid_size, block_size>>>(
          (const float *)a->data, (float *)output->data, num_elements);
      break;
    default:
      break;
    }
  } else {
    switch (a->dtype) {
    case FLOAT32:
      abs_cuda_forward_float_non_contig_kernel<<<grid_size, block_size>>>(
          (const float *)a->data, a->strides, (float *)output->data,
          output->strides, a->shape, a->ndim, num_elements);
      break;
    default:
      break;
    }
  }
  cudaDeviceSynchronize();
}

void abs_cuda_backward(Tensor **inputs, const Tensor *output, KernelParams params) {
  const Tensor *a = inputs[0];
  u64 num_elements = numel(a);

  int block_size = 256;
  int grid_size = (num_elements + block_size - 1) / block_size;

  if (is_contiguous(a) && is_contiguous(output)) {
    switch (a->dtype) {
    case FLOAT32:
      abs_cuda_backward_float_contig_kernel<<<grid_size, block_size>>>(
          (const float *)output->grad->data, (const float *)a->data,
          a->requires_grad ? (float *)a->grad->data : NULL, num_elements);
      break;
    default:
      break;
    }
  } else {
    switch (a->dtype) {
    case FLOAT32:
      abs_cuda_backward_float_non_contig_kernel<<<grid_size, block_size>>>(
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
  cudaDeviceSynchronize();
}
