#include "kernels/cuda/cuda_utils.cuh"
#include "kernels/mean.h"
#include "tensor.h"
#include <cuda_runtime.h>
#include <stdarg.h>

template <u64 block_size>
__device__ void warp_reduce(volatile float *sdata, u64 tid) {
  if (block_size >= 64)
    sdata[tid] += sdata[tid + 32];
  if (block_size >= 32)
    sdata[tid] += sdata[tid + 16];
  if (block_size >= 16)
    sdata[tid] += sdata[tid + 8];
  if (block_size >= 8)
    sdata[tid] += sdata[tid + 4];
  if (block_size >= 4)
    sdata[tid] += sdata[tid + 2];
  if (block_size >= 2)
    sdata[tid] += sdata[tid + 1];
}

template <u64 block_size>
__global__ void mean_cuda_forward_float_contig_kernel(const float *a, float *c,
                                                      u64 num_elements) {
  extern __shared__ float sdata[];
  u64 tid = threadIdx.x;
  u64 i = (blockDim.x * 2) * blockIdx.x + threadIdx.x;
  u64 grid_size = blockDim.x * 2 * gridDim.x;

  float sum_val = 0.0f;
  while (i < num_elements) {
    sum_val += a[i];
    if (i + blockDim.x < num_elements)
      sum_val += a[i + blockDim.x];
    i += grid_size;
  }
  sdata[tid] = sum_val;
  __syncthreads();

  for (u64 s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];

    __syncthreads();
  }

  if (tid < 32)
    warp_reduce<block_size>(sdata, tid);

  if (tid == 0)
    c[0] = sdata[0] / num_elements;
}

template <u64 block_size>
__global__ void mean_cuda_forward_float_noncontig_kernel(const float *a_data,
                                                         const u64 *a_shape,
                                                         const u64 *a_strides,
                                                         u64 a_ndim, float *c,
                                                         u64 num_elements) {
  extern __shared__ float sdata[];
  u64 tid = threadIdx.x;
  u64 i = (blockDim.x * 2) * blockIdx.x + threadIdx.x;
  u64 grid_size = blockDim.x * 2 * gridDim.x;

  u64 coords_i[MAX_NDIM];
  u64 coords_i_plus_blockdim[MAX_NDIM];

  float sum_val = 0.0f;
  while (i < num_elements) {
    float val1 = 0.0f;
    if (i < num_elements) {
      cuda_linear_to_coords(i, a_shape, a_ndim, coords_i);
      u64 offset_i = cuda_get_offset(coords_i, a_strides, a_ndim);
      val1 = a_data[offset_i];
    }

    float val2 = 0.0f;
    if (i + blockDim.x < num_elements) {
      cuda_linear_to_coords(i + blockDim.x, a_shape, a_ndim,
                            coords_i_plus_blockdim);
      u64 offset_i_plus_blockdim =
          cuda_get_offset(coords_i_plus_blockdim, a_strides, a_ndim);
      val2 = a_data[offset_i_plus_blockdim];
    }
    sum_val += val1 + val2;
    i += grid_size;
  }
  sdata[tid] = sum_val;
  __syncthreads();

  for (u64 s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  if (tid < 32)
    warp_reduce<block_size>(sdata, tid);

  if (tid == 0)
    c[0] = sdata[0] / num_elements;
}

__global__ void mean_cuda_forward_float_dim_contig_kernel(
    const float *a_data, const u64 *a_shape, const u64 *a_strides, u64 a_ndim,
    float *c_data, const u64 *c_shape, const u64 *c_strides, u64 c_ndim,
    u64 dim, u64 reduction_size, bool keepdim) {
  u64 output_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (output_idx >= cuda_numel_from_shape(c_shape, c_ndim)) {
    return;
  }

  u64 output_coords[MAX_NDIM];
  cuda_linear_to_coords(output_idx, c_shape, c_ndim, output_coords);

  float sum_val = 0.0f;
  u64 input_coords[MAX_NDIM];
  for (u64 i = 0; i < reduction_size; ++i) {
    for (u64 d = 0; d < a_ndim; ++d) {
      if (d == dim) {
        input_coords[d] = i;
      } else {
        input_coords[d] = output_coords[d > dim && !keepdim ? d - 1 : d];
      }
    }
    u64 input_offset = cuda_get_offset(input_coords, a_strides, a_ndim);
    sum_val += a_data[input_offset];
  }
  u64 output_offset = cuda_get_offset(output_coords, c_strides, c_ndim);
  c_data[output_offset] = sum_val / reduction_size;
}

__global__ void mean_cuda_forward_float_dim_noncontig_kernel(
    const float *a_data, const u64 *a_shape, const u64 *a_strides, u64 a_ndim,
    float *c_data, const u64 *c_shape, const u64 *c_strides, u64 c_ndim,
    u64 dim, u64 reduction_size, bool keepdim) {
  u64 output_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (output_idx >= cuda_numel_from_shape(c_shape, c_ndim)) {
    return;
  }

  u64 output_coords[MAX_NDIM];
  cuda_linear_to_coords(output_idx, c_shape, c_ndim, output_coords);

  float sum_val = 0.0f;
  u64 input_coords[MAX_NDIM];
  for (u64 i = 0; i < reduction_size; ++i) {
    for (u64 d = 0; d < a_ndim; ++d) {
      if (d == dim) {
        input_coords[d] = i;
      } else {
        input_coords[d] = output_coords[d > dim && !keepdim ? d - 1 : d];
      }
    }
    u64 input_offset = cuda_get_offset(input_coords, a_strides, a_ndim);
    sum_val += a_data[input_offset];
  }
  u64 output_offset = cuda_get_offset(output_coords, c_strides, c_ndim);
  c_data[output_offset] = sum_val / reduction_size;
}

__global__ void mean_cuda_backward_float_dim_contig_kernel(
    const float *dc_data, const u64 *dc_shape, const u64 *dc_strides,
    u64 dc_ndim, float *da_data, const u64 *da_shape, const u64 *da_strides,
    u64 da_ndim, u64 dim, u64 reduction_size, bool keepdim) {
  u64 input_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (input_idx >= cuda_numel_from_shape(da_shape, da_ndim)) {
    return;
  }

  u64 input_coords[MAX_NDIM];
  cuda_linear_to_coords(input_idx, da_shape, da_ndim, input_coords);

  u64 output_coords[MAX_NDIM];
  for (u64 i = 0; i < MAX_NDIM; ++i) {
    output_coords[i] = 0;
  }
  u64 output_dim_idx = 0;
  for (u64 d = 0; d < da_ndim; ++d) {
    if (d == dim) {
      if (keepdim) {
        output_coords[output_dim_idx++] = 0;
      } else {
        continue;
      }
    } else {
      output_coords[output_dim_idx++] = input_coords[d];
    }
  }
  u64 output_offset = cuda_get_offset(output_coords, dc_strides, dc_ndim);
  da_data[input_idx] += dc_data[output_offset] / reduction_size;
}

__global__ void mean_cuda_backward_float_dim_noncontig_kernel(
    const float *dc_data, const u64 *dc_shape, const u64 *dc_strides,
    u64 dc_ndim, float *da_data, const u64 *da_shape, const u64 *da_strides,
    u64 da_ndim, u64 dim, u64 reduction_size, bool keepdim) {
  u64 input_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (input_idx >= cuda_numel_from_shape(da_shape, da_ndim)) {
    return;
  }

  u64 input_coords[MAX_NDIM];
  cuda_linear_to_coords(input_idx, da_shape, da_ndim, input_coords);

  u64 output_coords[MAX_NDIM];
  for (u64 i = 0; i < MAX_NDIM; ++i) {
    output_coords[i] = 0;
  }
  u64 output_dim_idx = 0;
  for (u64 d = 0; d < da_ndim; ++d) {
    if (d == dim) {
      if (keepdim) {
        output_coords[output_dim_idx++] = 0;
      } else {
        continue;
      }
    } else {
      output_coords[output_dim_idx++] = input_coords[d];
    }
  }
  u64 output_offset = cuda_get_offset(output_coords, dc_strides, dc_ndim);
  u64 input_offset = cuda_get_offset(input_coords, da_strides, da_ndim);
  da_data[input_offset] += dc_data[output_offset] / reduction_size;
}

__global__ void mean_cuda_backward_float_contig_kernel(const float *dc,
                                                       float *da,
                                                       u64 num_elements) {
  u64 idx = blockDim.x * blockIdx.x + threadIdx.x;
  u64 strides = blockDim.x * gridDim.x;
  float grad = dc[0];

  for (u64 i = idx; i < num_elements; i += strides) {
    da[i] += grad / num_elements;
  }
}

__global__ void
mean_cuda_backward_float_noncontig_kernel(const float *dc, float *da,
                                          u64 *a_shape, u64 *a_strides,
                                          u64 a_ndim, u64 num_elements) {
  u64 idx = blockDim.x * blockIdx.x + threadIdx.x;
  u64 strides = blockDim.x * gridDim.x;

  u64 coords[MAX_NDIM];

  for (u64 i = idx; i < num_elements; i += strides) {
    cuda_linear_to_coords(i, a_shape, a_ndim, coords);
    u64 offset = cuda_get_offset(coords, a_strides, a_ndim);
    da[offset] += dc[0] / num_elements;
  }
}

void mean_cuda_forward(const Tensor **inputs, Tensor *output, ...) {
  const Tensor *a = inputs[0];
  float *c_data = (float *)output->data;

  va_list args;
  va_start(args, output);
  u64 dim = va_arg(args, u64);
  bool keepdim = va_arg(args, int);
  va_end(args);

  if (dim == MAX_NDIM + 1) {
    u64 num_elements = numel(a);
    const u64 block_size = 256;
    const u64 grid_size =
        (num_elements + (block_size * 2) - 1) / (block_size * 2);
    const u64 shared_mem_size = block_size * sizeof(float);

    if (is_contiguous(a)) {
      switch (a->dtype) {
      case FLOAT32:
        mean_cuda_forward_float_contig_kernel<block_size>
            <<<grid_size, block_size, shared_mem_size>>>((const float *)a->data,
                                                         c_data, num_elements);
        break;
      default:
        break;
      }
    } else {
      switch (a->dtype) {
      case FLOAT32:
        mean_cuda_forward_float_noncontig_kernel<block_size>
            <<<grid_size, block_size, shared_mem_size>>>(
                (const float *)a->data, a->shape, a->strides, a->ndim, c_data,
                num_elements);
        break;
      default:
        break;
      }
    }
  } else {
    u64 reduction_size = a->shape[dim];
    u64 output_num_elements = numel(output);

    const u64 block_size = 256;
    const u64 grid_size = (output_num_elements + block_size - 1) / block_size;

    if (is_contiguous(a)) {
      switch (a->dtype) {
      case FLOAT32:
        mean_cuda_forward_float_dim_contig_kernel<<<grid_size, block_size>>>(
            (const float *)a->data, a->shape, a->strides, a->ndim, c_data,
            output->shape, output->strides, output->ndim, dim, reduction_size,
            keepdim);
        break;
      default:
        break;
      }
    } else {
      switch (a->dtype) {
      case FLOAT32:
        mean_cuda_forward_float_dim_noncontig_kernel<<<grid_size, block_size>>>(
            (const float *)a->data, a->shape, a->strides, a->ndim, c_data,
            output->shape, output->strides, output->ndim, dim, reduction_size,
            keepdim);
        break;
      default:
        break;
      }
    }
  }
  cudaDeviceSynchronize();
}

void mean_cuda_backward(Tensor **inputs, const Tensor *output, ...) {
  Tensor *a = inputs[0];
  const float *dc_data = (const float *)output->grad->data;
  float *da_data = (float *)a->grad->data;

  va_list args;
  va_start(args, output);
  u64 dim = va_arg(args, u64);
  bool keepdim = va_arg(args, int);
  va_end(args);

  if (dim == MAX_NDIM + 1) {
    u64 num_elements = numel(a);
    const u64 block_size = 256;
    const u64 grid_size = (num_elements + block_size - 1) / block_size;

    if (is_contiguous(a)) {
      switch (a->dtype) {
      case FLOAT32:
        mean_cuda_backward_float_contig_kernel<<<grid_size, block_size>>>(
            dc_data, da_data, num_elements);
        break;
      default:
        break;
      }
    } else {
      switch (a->dtype) {
      case FLOAT32:
        mean_cuda_backward_float_noncontig_kernel<<<grid_size, block_size>>>(
            dc_data, da_data, a->shape, a->strides, a->ndim, num_elements);
        break;
      default:
        break;
      }
    }
  } else {
    u64 reduction_size = a->shape[dim];
    u64 input_num_elements = numel(a);

    const u64 block_size = 256;
    const u64 grid_size = (input_num_elements + block_size - 1) / block_size;

    if (is_contiguous(a)) {
      switch (a->dtype) {
      case FLOAT32:
        mean_cuda_backward_float_dim_contig_kernel<<<grid_size, block_size>>>(
            dc_data, output->shape, output->strides, output->ndim, da_data,
            a->shape, a->strides, a->ndim, dim, reduction_size, keepdim);
        break;
      default:
        break;
      }
    } else {
      switch (a->dtype) {
      case FLOAT32:
        mean_cuda_backward_float_dim_noncontig_kernel<<<grid_size,
                                                        block_size>>>(
            dc_data, output->shape, output->strides, output->ndim, da_data,
            a->shape, a->strides, a->ndim, dim, reduction_size, keepdim);
        break;
      default:
        break;
      }
    }
  }
}
