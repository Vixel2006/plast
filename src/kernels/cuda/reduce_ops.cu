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

template <typename Op>
__global__ void reduce_cuda_forward_float_dim_kernel(const float *a_data, const u64 *a_shape,
                                                     const u64 *a_strides, u64 a_ndim,
                                                     float *c_data, const u64 *c_shape,
                                                     const u64 *c_strides, u64 c_ndim, u64 dim,
                                                     u64 reduce_size, bool keepdim, Op op) {
  extern __shared__ float sdata[];
  u64 tid = threadIdx.x;
  u64 block_size = blockDim.x;
  u64 output_idx = blockIdx.x;

  u64 output_coords[MAX_NDIM];
  cuda_linear_to_coords(output_idx, c_shape, c_ndim, output_coords);

  u64 expanded[MAX_NDIM];
  for (u64 d = 0; d < a_ndim; ++d) {
    if (d == dim) {
      expanded[d] = 0;
    } else {
      expanded[d] = output_coords[d > dim && !keepdim ? d - 1 : d];
    }
  }
  u64 base = cuda_get_offset(expanded, a_strides, a_ndim);
  u64 stride = a_strides[dim];

  u64 chunk = (reduce_size + block_size - 1) / block_size;
  u64 start = tid * chunk;
  u64 end = min(start + chunk, reduce_size);

  float val = Op::identity();
  if (stride == 1) {
    for (u64 i = start; i < end; ++i)
      val = op(val, a_data[base + i]);
  } else {
    for (u64 i = start; i < end; ++i)
      val = op(val, a_data[base + i * stride]);
  }
  sdata[tid] = val;
  __syncthreads();

  for (u64 s = block_size / 2; s > 32; s >>= 1) {
    if (tid < s)
      sdata[tid] = op(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }

  if (tid < 32)
    warp_reduce<Op, 256>(sdata, tid, op);

  if (tid == 0) {
    u64 output_offset = cuda_get_offset(output_coords, c_strides, c_ndim);
    c_data[output_offset] = op.post_process(sdata[0], reduce_size);
  }
}

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

template <bool da_contig>
__global__ void
sum_mean_cuda_backward_float_dim_kernel(const float *dc_data, const u64 *dc_shape,
                                        const u64 *dc_strides, u64 dc_ndim, float *da_data,
                                        const u64 *da_shape, const u64 *da_strides, u64 da_ndim,
                                        u64 dim, u64 reduction_size, bool keepdim, bool is_mean) {
  u64 output_idx = blockIdx.x;
  u64 output_numel = cuda_numel_from_shape(dc_shape, dc_ndim);
  if (output_idx >= output_numel)
    return;
  u64 tid = threadIdx.x;

  u64 output_coords[MAX_NDIM];
  cuda_linear_to_coords(output_idx, dc_shape, dc_ndim, output_coords);
  u64 output_offset = cuda_get_offset(output_coords, dc_strides, dc_ndim);
  float grad = dc_data[output_offset];
  if (is_mean)
    grad /= reduction_size;

  u64 input_coords[MAX_NDIM];
  for (u64 d = 0; d < da_ndim; ++d) {
    if (d == dim) {
      input_coords[d] = 0;
    } else if (keepdim) {
      input_coords[d] = output_coords[d];
    } else {
      input_coords[d] = output_coords[d < dim ? d : d - 1];
    }
  }

  u64 base = cuda_get_offset(input_coords, da_strides, da_ndim);
  u64 stride = da_strides[dim];

  for (u64 r = tid; r < reduction_size; r += blockDim.x)
    da_data[base + r * stride] += grad;
}

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

template <bool da_contig>
__global__ void max_min_cuda_backward_float_dim_kernel(const float *dc_data, const u64 *dc_shape,
                                                       const u64 *dc_strides, u64 dc_ndim,
                                                       float *da_data, const float *a_data_fwd,
                                                       const float *c_data_fwd, const u64 *da_shape,
                                                       const u64 *da_strides, u64 da_ndim, u64 dim,
                                                       bool keepdim) {
  u64 output_idx = blockIdx.x;
  u64 output_numel = cuda_numel_from_shape(dc_shape, dc_ndim);
  if (output_idx >= output_numel)
    return;
  u64 tid = threadIdx.x;

  u64 output_coords[MAX_NDIM];
  cuda_linear_to_coords(output_idx, dc_shape, dc_ndim, output_coords);
  u64 output_offset = cuda_get_offset(output_coords, dc_strides, dc_ndim);
  float grad = dc_data[output_offset];
  float extreme = c_data_fwd[output_offset];

  u64 input_coords[MAX_NDIM];
  for (u64 d = 0; d < da_ndim; ++d) {
    if (d == dim) {
      input_coords[d] = 0;
    } else if (keepdim) {
      input_coords[d] = output_coords[d];
    } else {
      input_coords[d] = output_coords[d < dim ? d : d - 1];
    }
  }

  u64 base = cuda_get_offset(input_coords, da_strides, da_ndim);
  u64 stride = da_strides[dim];

  for (u64 r = tid; r < da_shape[dim]; r += blockDim.x) {
    u64 offset = base + r * stride;
    if (a_data_fwd[offset] == extreme)
      da_data[offset] += grad;
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
    const u64 shared_mem_size = block_size * sizeof(float);

    if (a->dtype == FLOAT32) {
      reduce_cuda_forward_float_dim_kernel<Op>
          <<<output_num_elements, block_size, shared_mem_size>>>(
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
    u64 output_num_elements = numel(output);

    const u64 block_size = 256;

    if (a->dtype == FLOAT32) {
      sum_mean_cuda_backward_float_dim_kernel<true><<<output_num_elements, block_size>>>(
          dc_data, output->shape, output->strides, output->ndim, da_data, a->shape, a->strides,
          a->ndim, dim, reduce_size, keepdim, is_mean);
    } else {
      fprintf(stderr, "plast: unsupported dtype %d\n", (int)a->dtype);
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
    u64 output_num_elements = numel(output);

    const u64 block_size = 256;

    if (a->dtype == FLOAT32) {
      max_min_cuda_backward_float_dim_kernel<true><<<output_num_elements, block_size>>>(
          dc_data, output->shape, output->strides, output->ndim, da_data, a_data_fwd, c_data_fwd,
          a->shape, a->strides, a->ndim, dim, keepdim);
    } else {
      fprintf(stderr, "plast: unsupported dtype %d\n", (int)a->dtype);
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

// =====================================================================
// Standalone benchmark (build with: make reduce_bench)
// =====================================================================
#ifdef REDUCE_BENCH

#include <math.h>
#include <stdlib.h>
#include <string.h>

extern "C" u64 numel(const Tensor *t) {
  return 0;
}
extern "C" bool is_contiguous(const Tensor *t) {
  return false;
}

#define BENCH_CUDA_CHECK(expr)                                                                     \
  do {                                                                                             \
    cudaError_t _err = (expr);                                                                     \
    if (_err != cudaSuccess) {                                                                     \
      fprintf(stderr, "CUDA error at %s:%d: '%s': %s\n", __FILE__, __LINE__, #expr,                \
              cudaGetErrorString(_err));                                                           \
      exit(1);                                                                                     \
    }                                                                                              \
  } while (0)

// ---- Block-size variants: instantiate for 64, 128, 256, 512 ----
#define BS_LIST 64, 128, 256, 512
#define BS_COUNT 4
static const int block_sizes[BS_COUNT] = {BS_LIST};

#define DEFINE_LAUNCHERS_BS(OpT, fn, bs)                                                           \
  static void launch_##fn##_contig_##bs(const float *a, float *c, u64 n) {                         \
    reduce_cuda_forward_float_contig_kernel<OpT, bs>                                               \
        <<<(n + bs * 2 - 1) / (bs * 2), bs, bs * sizeof(float)>>>(a, c, n, OpT());                 \
    BENCH_CUDA_CHECK(cudaDeviceSynchronize());                                                     \
  }                                                                                                \
  static void launch_##fn##_noncontig_##bs(const float *a, const u64 *sh, const u64 *st, u64 nd,   \
                                           float *c, u64 n) {                                      \
    reduce_cuda_forward_float_noncontig_kernel<OpT, bs>                                            \
        <<<(n + bs * 2 - 1) / (bs * 2), bs, bs * sizeof(float)>>>(a, sh, st, nd, c, n, OpT());     \
    BENCH_CUDA_CHECK(cudaDeviceSynchronize());                                                     \
  }

#define DEFINE_ALL_BS(OpT, fn)                                                                     \
  DEFINE_LAUNCHERS_BS(OpT, fn, 64)                                                                 \
  DEFINE_LAUNCHERS_BS(OpT, fn, 128)                                                                \
  DEFINE_LAUNCHERS_BS(OpT, fn, 256)                                                                \
  DEFINE_LAUNCHERS_BS(OpT, fn, 512)

DEFINE_ALL_BS(SumOp, sum)
DEFINE_ALL_BS(MeanOp, mean)
DEFINE_ALL_BS(MaxOp, max)
DEFINE_ALL_BS(MinOp, min)

typedef struct {
  const char *name;
  void (*contig[BS_COUNT])(const float *, float *, u64);
  void (*noncontig[BS_COUNT])(const float *, const u64 *, const u64 *, u64, float *, u64);
} OpBench;

#define FN_PTRS(fn)                                                                                \
  {launch_##fn##_contig_64, launch_##fn##_contig_128, launch_##fn##_contig_256,                    \
   launch_##fn##_contig_512},                                                                      \
  {                                                                                                \
    launch_##fn##_noncontig_64, launch_##fn##_noncontig_128, launch_##fn##_noncontig_256,          \
        launch_##fn##_noncontig_512                                                                \
  }

static const OpBench ops[] = {
    {"sum", FN_PTRS(sum)},
    {"mean", FN_PTRS(mean)},
    {"max", FN_PTRS(max)},
    {"min", FN_PTRS(min)},
};
static const int num_ops = 4;

static int bs_index(int block_size) {
  for (int i = 0; i < BS_COUNT; i++)
    if (block_sizes[i] == block_size)
      return i;
  fprintf(stderr, "Unsupported block size %d (use 64|128|256|512)\n", block_size);
  exit(1);
}

static void fill_rand(float *data, u64 n) {
  for (u64 i = 0; i < n; i++)
    data[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

// ---- Device setup shared by bench_config and profile_config ----
typedef struct {
  float *d_data;
  float *d_out;
  u64 *d_shape;
  u64 *d_strides;
} BenchBuf;

static BenchBuf bench_setup(u64 n, bool do_noncontig) {
  BenchBuf b = {NULL, NULL, NULL, NULL};
  u64 alloc_n = do_noncontig ? n * 2 : n;
  u64 bytes = alloc_n * sizeof(float);

  float *h_data = (float *)malloc(n * sizeof(float));
  fill_rand(h_data, n);
  BENCH_CUDA_CHECK(cudaMalloc(&b.d_data, bytes));
  BENCH_CUDA_CHECK(cudaMalloc(&b.d_out, sizeof(float)));

  if (do_noncontig) {
    float *h_strided = (float *)malloc(alloc_n * sizeof(float));
    for (u64 i = 0; i < alloc_n; i++)
      h_strided[i] = (i % 2 == 0) ? h_data[i / 2] : 0.0f;
    BENCH_CUDA_CHECK(cudaMemcpy(b.d_data, h_strided, bytes, cudaMemcpyHostToDevice));
    free(h_strided);
    u64 h_shape = n, h_stride = 2;
    BENCH_CUDA_CHECK(cudaMalloc(&b.d_shape, sizeof(u64)));
    BENCH_CUDA_CHECK(cudaMalloc(&b.d_strides, sizeof(u64)));
    BENCH_CUDA_CHECK(cudaMemcpy(b.d_shape, &h_shape, sizeof(u64), cudaMemcpyHostToDevice));
    BENCH_CUDA_CHECK(cudaMemcpy(b.d_strides, &h_stride, sizeof(u64), cudaMemcpyHostToDevice));
  } else {
    BENCH_CUDA_CHECK(cudaMemcpy(b.d_data, h_data, bytes, cudaMemcpyHostToDevice));
  }

  free(h_data);
  return b;
}

static void bench_teardown(BenchBuf *b) {
  cudaFree(b->d_data);
  cudaFree(b->d_out);
  cudaFree(b->d_shape);
  cudaFree(b->d_strides);
}

static void bench_run(u64 n, const OpBench *op, bool do_noncontig, int bs_idx, int warmup,
                      int trials) {
  BenchBuf b = bench_setup(n, do_noncontig);

  for (int i = 0; i < warmup; i++) {
    if (do_noncontig)
      op->noncontig[bs_idx](b.d_data, b.d_shape, b.d_strides, 1, b.d_out, n);
    else
      op->contig[bs_idx](b.d_data, b.d_out, n);
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < trials; i++) {
    if (do_noncontig)
      op->noncontig[bs_idx](b.d_data, b.d_shape, b.d_strides, 1, b.d_out, n);
    else
      op->contig[bs_idx](b.d_data, b.d_out, n);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_ms = ms / trials;
  double bw = (double)n * sizeof(float) / (avg_ms * 1e-3) / 1e9;
  int bs = block_sizes[bs_idx];

  printf("| %-6s | %-12s | %4d | %9lu | %8.3f | %8.2f |\n", op->name,
         do_noncontig ? "noncontig" : "contiguous", bs, n, avg_ms, bw);

  bench_teardown(&b);
}

// ---- Sizes to sweep ----
static const u64 bench_sizes[] = {1024,    4096,    16384,    65536,   262144,
                                  1048576, 4194304, 16777216, 67108864};
static const int num_bench_sizes = sizeof(bench_sizes) / sizeof(bench_sizes[0]);

static void bench_all(int warmup, int trials) {
  printf("+--------+--------------+------+-----------+----------+----------+\n");
  printf("| Op     | Layout       |  BS  |   Elements| Time(ms) | BW(GB/s) |\n");
  printf("+--------+--------------+------+-----------+----------+----------+\n");

  for (int o = 0; o < num_ops; o++) {
    for (int s = 0; s < num_bench_sizes; s++) {
      u64 n = bench_sizes[s];
      if (n > 64 * 1024 * 1024)
        break;
      // default block size = 256 (index 2)
      bench_run(n, &ops[o], false, 2, warmup, trials);
      bench_run(n, &ops[o], true, 2, warmup, trials);
    }
  }

  printf("+--------+--------------+------+-----------+----------+----------+\n");
}

// ---- Dim reduction benchmarking (2D) ----
template <typename Op>
static void launch_dim_tpl(const float *a_data, const u64 *a_shape, const u64 *a_strides,
                           u64 a_ndim, float *c_data, const u64 *c_shape, const u64 *c_strides,
                           u64 c_ndim, u64 dim, u64 reduce_size, bool keepdim, u64 output_numel) {
  reduce_cuda_forward_float_dim_kernel<Op><<<output_numel, 256, 256 * sizeof(float)>>>(
      a_data, a_shape, a_strides, a_ndim, c_data, c_shape, c_strides, c_ndim, dim, reduce_size,
      keepdim, Op());
  BENCH_CUDA_CHECK(cudaDeviceSynchronize());
}

#define DEFINE_DIM_LAUNCHER(OpT, fn)                                                               \
  static void launch_dim_##fn(const float *a_data, const u64 *a_shape, const u64 *a_strides,       \
                              u64 a_ndim, float *c_data, const u64 *c_shape, const u64 *c_strides, \
                              u64 c_ndim, u64 dim, u64 reduce_size, bool keepdim,                  \
                              u64 output_numel) {                                                  \
    launch_dim_tpl<OpT>(a_data, a_shape, a_strides, a_ndim, c_data, c_shape, c_strides, c_ndim,    \
                        dim, reduce_size, keepdim, output_numel);                                  \
  }

DEFINE_DIM_LAUNCHER(SumOp, sum)
DEFINE_DIM_LAUNCHER(MeanOp, mean)
DEFINE_DIM_LAUNCHER(MaxOp, max)
DEFINE_DIM_LAUNCHER(MinOp, min)

typedef void (*dim_launcher_t)(const float *, const u64 *, const u64 *, u64, float *, const u64 *,
                               const u64 *, u64, u64, u64, bool, u64);

static dim_launcher_t get_dim_launcher(const char *name) {
  if (strcmp(name, "sum") == 0)
    return launch_dim_sum;
  if (strcmp(name, "mean") == 0)
    return launch_dim_mean;
  if (strcmp(name, "max") == 0)
    return launch_dim_max;
  if (strcmp(name, "min") == 0)
    return launch_dim_min;
  return NULL;
}

static void bench_dim_config(u64 rows, u64 cols, u64 dim, const char *op_name, int warmup,
                             int trials) {
  dim_launcher_t launch = get_dim_launcher(op_name);
  if (!launch) {
    fprintf(stderr, "bad op: %s\n", op_name);
    return;
  }

  bool keepdim = false;
  u64 a_shape_h[2] = {rows, cols};
  u64 a_strides_h[2] = {cols, 1};
  u64 a_ndim = 2;
  u64 reduce_size = a_shape_h[dim];
  u64 c_ndim = keepdim ? 2 : 1;
  u64 c_shape_h[2], c_strides_h[2];
  u64 output_numel = 1;
  u64 ci = 0;
  for (u64 d = 0; d < a_ndim; ++d) {
    if (d == dim) {
      if (keepdim) {
        c_shape_h[ci] = 1;
        c_strides_h[ci] = 0;
        ci++;
      }
    } else {
      c_shape_h[ci] = a_shape_h[d];
      c_strides_h[ci] = 1;
      output_numel *= c_shape_h[ci];
      ci++;
    }
  }
  // When keepdim=false and there's only one non-reduced dim, stride is 1
  c_strides_h[0] = 1;

  u64 *d_a_shape, *d_a_strides, *d_c_shape, *d_c_strides;
  float *d_a, *d_c;
  u64 a_bytes = rows * cols * sizeof(float);

  float *h_a = (float *)malloc(a_bytes);
  fill_rand(h_a, rows * cols);

  BENCH_CUDA_CHECK(cudaMalloc(&d_a, a_bytes));
  BENCH_CUDA_CHECK(cudaMemcpy(d_a, h_a, a_bytes, cudaMemcpyHostToDevice));
  BENCH_CUDA_CHECK(cudaMalloc(&d_c, output_numel * sizeof(float)));
  BENCH_CUDA_CHECK(cudaMalloc(&d_a_shape, 2 * sizeof(u64)));
  BENCH_CUDA_CHECK(cudaMalloc(&d_a_strides, 2 * sizeof(u64)));
  BENCH_CUDA_CHECK(cudaMalloc(&d_c_shape, 2 * sizeof(u64)));
  BENCH_CUDA_CHECK(cudaMalloc(&d_c_strides, 2 * sizeof(u64)));
  BENCH_CUDA_CHECK(cudaMemcpy(d_a_shape, a_shape_h, 2 * sizeof(u64), cudaMemcpyHostToDevice));
  BENCH_CUDA_CHECK(cudaMemcpy(d_a_strides, a_strides_h, 2 * sizeof(u64), cudaMemcpyHostToDevice));
  BENCH_CUDA_CHECK(cudaMemcpy(d_c_shape, c_shape_h, c_ndim * sizeof(u64), cudaMemcpyHostToDevice));
  BENCH_CUDA_CHECK(
      cudaMemcpy(d_c_strides, c_strides_h, c_ndim * sizeof(u64), cudaMemcpyHostToDevice));

  for (int i = 0; i < warmup; i++)
    launch(d_a, d_a_shape, d_a_strides, a_ndim, d_c, d_c_shape, d_c_strides, c_ndim, dim,
           reduce_size, keepdim, output_numel);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < trials; i++)
    launch(d_a, d_a_shape, d_a_strides, a_ndim, d_c, d_c_shape, d_c_strides, c_ndim, dim,
           reduce_size, keepdim, output_numel);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_ms = ms / trials;
  u64 total_elements = rows * cols;
  double bw = (double)total_elements * sizeof(float) / (avg_ms * 1e-3) / 1e9;
  const char *dim_name = (dim == 0) ? "rows" : "cols";

  printf("| %-6s | dim-%-5s | %4d | %4lux%-4lu | %8.3f | %8.2f |\n", op_name, dim_name, 256, rows,
         cols, avg_ms, bw);

  free(h_a);
  cudaFree(d_a);
  cudaFree(d_c);
  cudaFree(d_a_shape);
  cudaFree(d_a_strides);
  cudaFree(d_c_shape);
  cudaFree(d_c_strides);
}

// ---- Profile mode: single clean launch for ncu ----
static void profile_run(u64 n, const OpBench *op, bool do_noncontig, int bs_idx) {
  BenchBuf b = bench_setup(n, do_noncontig);
  int bs = block_sizes[bs_idx];

  printf("\n--- ncu profile target ---\n");
  printf("  kernel: reduce_cuda_forward_float_%s_kernel<%s,%d>\n",
         do_noncontig ? "noncontig" : "contig", op->name, bs);
  printf("  op: %s  |  layout: %s  |  elements: %lu  |  block_size: %d\n", op->name,
         do_noncontig ? "noncontig" : "contiguous", n, bs);
  printf("---------------------------\n\n");

  if (do_noncontig)
    op->noncontig[bs_idx](b.d_data, b.d_shape, b.d_strides, 1, b.d_out, n);
  else
    op->contig[bs_idx](b.d_data, b.d_out, n);

  bench_teardown(&b);
}

int main(int argc, char **argv) {
  int warmup = 5;
  int trials = 10;
  u64 single_n = 0;
  const char *single_op = NULL;
  bool single_noncontig = false;
  int block_size = 256;
  bool profile_mode = false;
  bool sweep_mode = true;

  // dim-reduction mode
  u64 dim_rows = 0, dim_cols = 0;
  u64 dim_dim = 1;
  bool dim_mode = false;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
      warmup = atoi(argv[++i]);
    else if (strcmp(argv[i], "--trials") == 0 && i + 1 < argc)
      trials = atoi(argv[++i]);
    else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
      single_n = (u64)atol(argv[++i]);
      sweep_mode = false;
    } else if (strcmp(argv[i], "--op") == 0 && i + 1 < argc) {
      single_op = argv[++i];
      sweep_mode = false;
    } else if (strcmp(argv[i], "--noncontig") == 0) {
      single_noncontig = true;
      sweep_mode = false;
    } else if (strcmp(argv[i], "--block-size") == 0 && i + 1 < argc) {
      block_size = atoi(argv[++i]);
      sweep_mode = false;
    } else if (strcmp(argv[i], "--profile") == 0) {
      profile_mode = true;
      sweep_mode = false;
    } else if (strcmp(argv[i], "--rows") == 0 && i + 1 < argc) {
      dim_rows = (u64)atol(argv[++i]);
      dim_mode = true;
      sweep_mode = false;
    } else if (strcmp(argv[i], "--cols") == 0 && i + 1 < argc) {
      dim_cols = (u64)atol(argv[++i]);
      dim_mode = true;
      sweep_mode = false;
    } else if (strcmp(argv[i], "--dim") == 0 && i + 1 < argc) {
      dim_dim = (u64)atol(argv[++i]);
      dim_mode = true;
      sweep_mode = false;
    } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      printf("Usage: reduce_bench [options]\n");
      printf("\nGlobal reduction:\n");
      printf("  (no args)        Full sweep (ops × sizes, block_size=256)\n");
      printf("  --op NAME        sum|mean|max|min\n");
      printf("  --size N         Number of elements\n");
      printf("  --block-size N   Threads per block: 64|128|256|512 (default: 256)\n");
      printf("  --noncontig      Strided layout (default: contiguous)\n");
      printf("  --profile        Single clean launch for ncu\n");
      printf("  --warmup N       Warmup iterations (default: 5)\n");
      printf("  --trials N       Timed iterations (default: 10)\n");
      printf("\nDim-wise reduction:\n");
      printf("  --rows M         Number of rows in 2D input\n");
      printf("  --cols N         Number of cols in 2D input\n");
      printf("  --dim D          0=reduce rows, 1=reduce cols (default: 1)\n");
      printf("  --help, -h       This help\n");
      return 0;
    } else {
      fprintf(stderr, "Unknown option: %s\n", argv[i]);
      return 1;
    }
  }

  int dev;
  BENCH_CUDA_CHECK(cudaGetDevice(&dev));
  struct cudaDeviceProp prop;
  BENCH_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

  if (!profile_mode)
    printf("Device: %s (SM %d.%d, %d SMs, %.0f GB memory)\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, (double)prop.totalGlobalMem / 1e9);

  srand(42);

  if (dim_mode) {
    if (!single_op)
      single_op = "sum";
    if (dim_rows == 0)
      dim_rows = 4096;
    if (dim_cols == 0)
      dim_cols = 4096;
    printf("+--------+----------+------+----------+----------+----------+\n");
    printf("| Op     | Mode     |  BS  | Shape    | Time(ms) | BW(GB/s) |\n");
    printf("+--------+----------+------+----------+----------+----------+\n");
    bench_dim_config(dim_rows, dim_cols, dim_dim, single_op, warmup, trials);
    printf("+--------+----------+------+----------+----------+----------+\n");
    return 0;
  }

  int bs_idx = bs_index(block_size);

  if (sweep_mode) {
    bench_all(warmup, trials);
    return 0;
  }

  const OpBench *selected_op = NULL;
  for (int i = 0; i < num_ops; i++) {
    if (strcmp(ops[i].name, single_op) == 0) {
      selected_op = &ops[i];
      break;
    }
  }
  if (single_op && !selected_op) {
    fprintf(stderr, "Unknown op: %s (use sum|mean|max|min)\n", single_op);
    return 1;
  }

  u64 n = single_n ? single_n : (1024 * 1024);

  if (profile_mode) {
    if (!selected_op)
      selected_op = &ops[0];
    profile_run(n, selected_op, single_noncontig, bs_idx);
    return 0;
  }

  if (selected_op) {
    bench_run(n, selected_op, single_noncontig, bs_idx, warmup, trials);
    return 0;
  }

  return 0;
}

#endif // REDUCE_BENCH
