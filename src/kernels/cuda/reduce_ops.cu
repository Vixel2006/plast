#include "kernels/ops/reduce.h"
#include "kernels/cuda/cuda_utils.cuh"
#include "kernels/cuda/cuda_check.h"
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>

// Functors
struct SumOp {
  __device__ __forceinline__ float operator()(float a, float b) const {
    return a + b;
  }
  __device__ __forceinline__ static float identity() {
    return 0.0f;
  }
  __device__ __forceinline__ float post_process(float val, u64 n) const {
    return val;
  }
};

struct MeanOp {
  __device__ __forceinline__ float operator()(float a, float b) const {
    return a + b;
  }
  __device__ __forceinline__ static float identity() {
    return 0.0f;
  }
  __device__ __forceinline__ float post_process(float val, u64 n) const {
    return val / n;
  }
};

struct MaxOp {
  __device__ __forceinline__ float operator()(float a, float b) const {
    return fmaxf(a, b);
  }
  __device__ __forceinline__ static float identity() {
    return -FLT_MAX;
  }
  __device__ __forceinline__ float post_process(float val, u64 n) const {
    return val;
  }
};

struct MinOp {
  __device__ __forceinline__ float operator()(float a, float b) const {
    return fminf(a, b);
  }
  __device__ __forceinline__ static float identity() {
    return FLT_MAX;
  }
  __device__ __forceinline__ float post_process(float val, u64 n) const {
    return val;
  }
};

// Device helper: warp reduction
template <typename Op, u64 block_size>
__device__ __forceinline__ void warp_reduce(volatile float *sdata, u64 tid, Op op) {
  if (block_size >= 64)
    sdata[tid] = op(sdata[tid], sdata[tid + 32]);
  if (block_size >= 32)
    sdata[tid] = op(sdata[tid], sdata[tid + 16]);
  if (block_size >= 16)
    sdata[tid] = op(sdata[tid], sdata[tid + 8]);
  if (block_size >= 8)
    sdata[tid] = op(sdata[tid], sdata[tid + 4]);
  if (block_size >= 4)
    sdata[tid] = op(sdata[tid], sdata[tid + 2]);
  if (block_size >= 2)
    sdata[tid] = op(sdata[tid], sdata[tid + 1]);
}

// Global reduction kernels (forward)
template <typename Op, u64 block_size>
__global__ void reduce_cuda_forward_float_contig_kernel(const float *a, float *c, u64 num_elements,
                                                        Op op) {
  extern __shared__ float sdata[];
  u64 tid = threadIdx.x;
  u64 i = (blockDim.x * 2) * blockIdx.x + threadIdx.x;
  u64 grid_size = blockDim.x * 2 * gridDim.x;

  float thread_val = Op::identity();
  while (i < num_elements) {
    thread_val = op(thread_val, a[i]);
    if (i + blockDim.x < num_elements)
      thread_val = op(thread_val, a[i + blockDim.x]);
    i += grid_size;
  }
  sdata[tid] = thread_val;
  __syncthreads();

  for (u64 s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s)
      sdata[tid] = op(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }

  if (tid < 32)
    warp_reduce<Op, block_size>(sdata, tid, op);

  if (tid == 0)
    c[0] = op.post_process(sdata[0], num_elements);
}

template <typename Op, u64 block_size>
__global__ void reduce_cuda_forward_float_noncontig_kernel(const float *a_data, const u64 *a_shape,
                                                           const u64 *a_strides, u64 a_ndim,
                                                           float *c, u64 num_elements, Op op) {
  extern __shared__ float sdata[];
  u64 tid = threadIdx.x;
  u64 i = (blockDim.x * 2) * blockIdx.x + threadIdx.x;
  u64 grid_size = blockDim.x * 2 * gridDim.x;

  u64 coords_i[MAX_NDIM];
  u64 coords_i_plus_blockdim[MAX_NDIM];

  float thread_val = Op::identity();
  while (i < num_elements) {
    float val1 = Op::identity();
    if (i < num_elements) {
      cuda_linear_to_coords(i, a_shape, a_ndim, coords_i);
      u64 offset_i = cuda_get_offset(coords_i, a_strides, a_ndim);
      val1 = a_data[offset_i];
    }

    float val2 = Op::identity();
    if (i + blockDim.x < num_elements) {
      cuda_linear_to_coords(i + blockDim.x, a_shape, a_ndim, coords_i_plus_blockdim);
      u64 offset_i_plus_blockdim = cuda_get_offset(coords_i_plus_blockdim, a_strides, a_ndim);
      val2 = a_data[offset_i_plus_blockdim];
    }
    thread_val = op(thread_val, op(val1, val2));
    i += grid_size;
  }
  sdata[tid] = thread_val;
  __syncthreads();

  for (u64 s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s)
      sdata[tid] = op(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }

  if (tid < 32)
    warp_reduce<Op, block_size>(sdata, tid, op);

  if (tid == 0)
    c[0] = op.post_process(sdata[0], num_elements);
}

// Dimension-wise reduction kernel (forward)
template <typename Op>
__global__ void reduce_cuda_forward_float_dim_kernel(const float *a_data, const u64 *a_shape,
                                                     const u64 *a_strides, u64 a_ndim,
                                                     float *c_data, const u64 *c_shape,
                                                     const u64 *c_strides, u64 c_ndim, u64 dim,
                                                     u64 reduce_size, bool keepdim, Op op) {
  u64 output_idx = blockIdx.x * blockDim.x + threadIdx.x;
  u64 output_numel = cuda_numel_from_shape(c_shape, c_ndim);
  if (output_idx >= output_numel) {
    return;
  }

  u64 output_coords[MAX_NDIM];
  cuda_linear_to_coords(output_idx, c_shape, c_ndim, output_coords);

  float val = Op::identity();
  u64 input_coords[MAX_NDIM];
  for (u64 i = 0; i < reduce_size; ++i) {
    for (u64 d = 0; d < a_ndim; ++d) {
      if (d == dim) {
        input_coords[d] = i;
      } else {
        input_coords[d] = output_coords[d > dim && !keepdim ? d - 1 : d];
      }
    }
    u64 input_offset = cuda_get_offset(input_coords, a_strides, a_ndim);
    val = op(val, a_data[input_offset]);
  }
  u64 output_offset = cuda_get_offset(output_coords, c_strides, c_ndim);
  c_data[output_offset] = op.post_process(val, reduce_size);
}

// Global sum/mean backward kernel
template <bool da_contig>
__global__ void sum_mean_cuda_backward_float_kernel(const float *dc, float *da, const u64 *da_shape,
                                                    const u64 *da_strides, u64 da_ndim,
                                                    u64 num_elements, bool is_mean) {
  u64 idx = blockDim.x * blockIdx.x + threadIdx.x;
  u64 strides = blockDim.x * gridDim.x;
  float grad = dc[0];
  if (is_mean) {
    grad /= num_elements;
  }

  if (da_contig) {
    for (u64 i = idx; i < num_elements; i += strides) {
      da[i] += grad;
    }
  } else {
    u64 coords[MAX_NDIM];
    for (u64 i = idx; i < num_elements; i += strides) {
      cuda_linear_to_coords(i, da_shape, da_ndim, coords);
      u64 offset = cuda_get_offset(coords, da_strides, da_ndim);
      da[offset] += grad;
    }
  }
}

// Dimension sum/mean backward kernel
template <bool da_contig>
__global__ void
sum_mean_cuda_backward_float_dim_kernel(const float *dc_data, const u64 *dc_shape,
                                        const u64 *dc_strides, u64 dc_ndim, float *da_data,
                                        const u64 *da_shape, const u64 *da_strides, u64 da_ndim,
                                        u64 dim, u64 reduction_size, bool keepdim, bool is_mean) {
  u64 input_idx = blockIdx.x * blockDim.x + threadIdx.x;
  u64 input_numel = cuda_numel_from_shape(da_shape, da_ndim);
  if (input_idx >= input_numel) {
    return;
  }

  u64 input_coords[MAX_NDIM];
  cuda_linear_to_coords(input_idx, da_shape, da_ndim, input_coords);

  u64 output_coords[MAX_NDIM];
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
  float grad = dc_data[output_offset];
  if (is_mean) {
    grad /= reduction_size;
  }

  if (da_contig) {
    da_data[input_idx] += grad;
  } else {
    u64 input_offset = cuda_get_offset(input_coords, da_strides, da_ndim);
    da_data[input_offset] += grad;
  }
}

// Global max/min backward kernel
template <bool da_contig>
__global__ void max_min_cuda_backward_float_kernel(const float *dc, float *da,
                                                   const float *a_data_fwd, const float *c_data_fwd,
                                                   const u64 *da_shape, const u64 *da_strides,
                                                   u64 da_ndim, u64 num_elements) {
  u64 idx = blockDim.x * blockIdx.x + threadIdx.x;
  u64 strides = blockDim.x * gridDim.x;
  float grad = dc[0];
  float extreme_val = c_data_fwd[0];

  if (da_contig) {
    for (u64 i = idx; i < num_elements; i += strides) {
      if (a_data_fwd[i] == extreme_val) {
        da[i] += grad;
      }
    }
  } else {
    u64 coords[MAX_NDIM];
    for (u64 i = idx; i < num_elements; i += strides) {
      cuda_linear_to_coords(i, da_shape, da_ndim, coords);
      u64 offset = cuda_get_offset(coords, da_strides, da_ndim);
      if (a_data_fwd[offset] == extreme_val) {
        da[offset] += grad;
      }
    }
  }
}

// Dimension max/min backward kernel
template <bool da_contig>
__global__ void max_min_cuda_backward_float_dim_kernel(const float *dc_data, const u64 *dc_shape,
                                                       const u64 *dc_strides, u64 dc_ndim,
                                                       float *da_data, const float *a_data_fwd,
                                                       const float *c_data_fwd, const u64 *da_shape,
                                                       const u64 *da_strides, u64 da_ndim, u64 dim,
                                                       bool keepdim) {
  u64 input_idx = blockIdx.x * blockDim.x + threadIdx.x;
  u64 input_numel = cuda_numel_from_shape(da_shape, da_ndim);
  if (input_idx >= input_numel) {
    return;
  }

  u64 input_coords[MAX_NDIM];
  cuda_linear_to_coords(input_idx, da_shape, da_ndim, input_coords);

  u64 input_offset;
  if (da_contig) {
    input_offset = input_idx;
  } else {
    input_offset = cuda_get_offset(input_coords, da_strides, da_ndim);
  }

  u64 output_coords[MAX_NDIM];
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

  if (a_data_fwd[input_offset] == c_data_fwd[output_offset]) {
    da_data[input_offset] += dc_data[output_offset];
  }
}

// Forward Host Dispatcher Template
template <typename Op>
void reduce_cuda_forward_impl(const Tensor **inputs, Tensor *output, KernelParams params, Op op) {
  const Tensor *a = inputs[0];
  float *c_data = (float *)output->data;

  u64 dim = params.dim;
  bool keepdim = params.keepdim;

  if (dim == MAX_NDIM + 1) {
    u64 num_elements = numel(a);
    const u64 block_size = 256;
    const u64 grid_size = (num_elements + (block_size * 2) - 1) / (block_size * 2);
    const u64 shared_mem_size = block_size * sizeof(float);

    if (is_contiguous(a)) {
      if (a->dtype == FLOAT32) {
        reduce_cuda_forward_float_contig_kernel<Op, block_size>
            <<<grid_size, block_size, shared_mem_size>>>((const float *)a->data, c_data,
                                                         num_elements, op);
      } else {
        fprintf(stderr, "plast: unsupported dtype %d\n", (int)a->dtype);
      }
    } else {
      if (a->dtype == FLOAT32) {
        reduce_cuda_forward_float_noncontig_kernel<Op, block_size>
            <<<grid_size, block_size, shared_mem_size>>>(
                (const float *)a->data, a->shape, a->strides, a->ndim, c_data, num_elements, op);
      } else {
        fprintf(stderr, "plast: unsupported dtype %d\n", (int)a->dtype);
      }
    }
  } else {
    u64 reduce_size = a->shape[dim];
    u64 output_num_elements = numel(output);

    const u64 block_size = 256;
    const u64 grid_size = (output_num_elements + block_size - 1) / block_size;

    if (a->dtype == FLOAT32) {
      reduce_cuda_forward_float_dim_kernel<Op><<<grid_size, block_size>>>(
          (const float *)a->data, a->shape, a->strides, a->ndim, c_data, output->shape,
          output->strides, output->ndim, dim, reduce_size, keepdim, op);
    } else {
      fprintf(stderr, "plast: unsupported dtype %d\n", (int)a->dtype);
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());
}

// Forward host function exports
extern "C" void sum_cuda_forward(const Tensor **inputs, Tensor *output, KernelParams params) {
  reduce_cuda_forward_impl(inputs, output, params, SumOp());
}

extern "C" void mean_cuda_forward(const Tensor **inputs, Tensor *output, KernelParams params) {
  reduce_cuda_forward_impl(inputs, output, params, MeanOp());
}

extern "C" void max_cuda_forward(const Tensor **inputs, Tensor *output, KernelParams params) {
  reduce_cuda_forward_impl(inputs, output, params, MaxOp());
}

extern "C" void min_cuda_forward(const Tensor **inputs, Tensor *output, KernelParams params) {
  reduce_cuda_forward_impl(inputs, output, params, MinOp());
}

// Backward Host Dispatcher Helpers
void sum_mean_cuda_backward_impl(Tensor **inputs, const Tensor *output, KernelParams params,
                                 bool is_mean) {
  Tensor *a = inputs[0];
  const float *dc_data = (const float *)output->grad->data;
  float *da_data = (float *)a->grad->data;

  u64 dim = params.dim;
  bool keepdim = params.keepdim;

  if (dim == MAX_NDIM + 1) {
    u64 num_elements = numel(a);
    const u64 block_size = 256;
    const u64 grid_size = (num_elements + block_size - 1) / block_size;

    if (is_contiguous(a)) {
      if (a->dtype == FLOAT32) {
        sum_mean_cuda_backward_float_kernel<true><<<grid_size, block_size>>>(
            dc_data, da_data, a->shape, a->strides, a->ndim, num_elements, is_mean);
      } else {
        fprintf(stderr, "plast: unsupported dtype %d\n", (int)a->dtype);
      }
    } else {
      if (a->dtype == FLOAT32) {
        sum_mean_cuda_backward_float_kernel<false><<<grid_size, block_size>>>(
            dc_data, da_data, a->shape, a->strides, a->ndim, num_elements, is_mean);
      } else {
        fprintf(stderr, "plast: unsupported dtype %d\n", (int)a->dtype);
      }
    }
  } else {
    u64 reduce_size = a->shape[dim];
    u64 input_num_elements = numel(a);

    const u64 block_size = 256;
    const u64 grid_size = (input_num_elements + block_size - 1) / block_size;

    if (is_contiguous(a)) {
      if (a->dtype == FLOAT32) {
        sum_mean_cuda_backward_float_dim_kernel<true><<<grid_size, block_size>>>(
            dc_data, output->shape, output->strides, output->ndim, da_data, a->shape, a->strides,
            a->ndim, dim, reduce_size, keepdim, is_mean);
      } else {
        fprintf(stderr, "plast: unsupported dtype %d\n", (int)a->dtype);
      }
    } else {
      if (a->dtype == FLOAT32) {
        sum_mean_cuda_backward_float_dim_kernel<false><<<grid_size, block_size>>>(
            dc_data, output->shape, output->strides, output->ndim, da_data, a->shape, a->strides,
            a->ndim, dim, reduce_size, keepdim, is_mean);
      } else {
        fprintf(stderr, "plast: unsupported dtype %d\n", (int)a->dtype);
      }
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());
}

void max_min_cuda_backward_impl(Tensor **inputs, const Tensor *output, KernelParams params) {
  Tensor *a = inputs[0];
  const float *dc_data = (const float *)output->grad->data;
  float *da_data = (float *)a->grad->data;
  u64 dim = params.dim;
  bool keepdim = params.keepdim;

  const float *a_data_fwd = (const float *)a->data;
  const float *c_data_fwd = (const float *)output->data;

  if (dim == MAX_NDIM + 1) {
    u64 num_elements = numel(a);
    const u64 block_size = 256;
    const u64 grid_size = (num_elements + block_size - 1) / block_size;

    if (is_contiguous(a)) {
      if (a->dtype == FLOAT32) {
        max_min_cuda_backward_float_kernel<true><<<grid_size, block_size>>>(
            dc_data, da_data, a_data_fwd, c_data_fwd, a->shape, a->strides, a->ndim, num_elements);
      } else {
        fprintf(stderr, "plast: unsupported dtype %d\n", (int)a->dtype);
      }
    } else {
      if (a->dtype == FLOAT32) {
        max_min_cuda_backward_float_kernel<false><<<grid_size, block_size>>>(
            dc_data, da_data, a_data_fwd, c_data_fwd, a->shape, a->strides, a->ndim, num_elements);
      } else {
        fprintf(stderr, "plast: unsupported dtype %d\n", (int)a->dtype);
      }
    }
  } else {
    u64 input_num_elements = numel(a);

    const u64 block_size = 256;
    const u64 grid_size = (input_num_elements + block_size - 1) / block_size;

    if (is_contiguous(a)) {
      if (a->dtype == FLOAT32) {
        max_min_cuda_backward_float_dim_kernel<true><<<grid_size, block_size>>>(
            dc_data, output->shape, output->strides, output->ndim, da_data, a_data_fwd, c_data_fwd,
            a->shape, a->strides, a->ndim, dim, keepdim);
      } else {
        fprintf(stderr, "plast: unsupported dtype %d\n", (int)a->dtype);
      }
    } else {
      if (a->dtype == FLOAT32) {
        max_min_cuda_backward_float_dim_kernel<false><<<grid_size, block_size>>>(
            dc_data, output->shape, output->strides, output->ndim, da_data, a_data_fwd, c_data_fwd,
            a->shape, a->strides, a->ndim, dim, keepdim);
      } else {
        fprintf(stderr, "plast: unsupported dtype %d\n", (int)a->dtype);
      }
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());
}

// Backward host function exports
extern "C" void sum_cuda_backward(Tensor **inputs, const Tensor *output, KernelParams params) {
  sum_mean_cuda_backward_impl(inputs, output, params, false);
}

extern "C" void mean_cuda_backward(Tensor **inputs, const Tensor *output, KernelParams params) {
  sum_mean_cuda_backward_impl(inputs, output, params, true);
}

extern "C" void max_cuda_backward(Tensor **inputs, const Tensor *output, KernelParams params) {
  max_min_cuda_backward_impl(inputs, output, params);
}

extern "C" void min_cuda_backward(Tensor **inputs, const Tensor *output, KernelParams params) {
  max_min_cuda_backward_impl(inputs, output, params);
}
