#include "cuda_check.h"
#include "core/definitions.h"
#include "kernels/pack.h"
#include "kernels/ops/shape.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / b)

#define TM 8
#define TN 8
#define BK 16
#define BM 128
#define BN 128

__global__ void matmul_cuda_forward_contig_kernel(const float *a, const float *b, float *c,
                                                  u64 batches, u64 rows, u64 inners, u64 cols) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int bz = blockIdx.z;

  __shared__ float a_shared[BM][BK + 1];
  __shared__ float b_shared[BK][BN];

  float c_reg[TM][TN] = {0.0f};

  long long a_base = (long long)bz * rows * inners;
  long long b_base = (long long)bz * inners * cols;
  long long c_base = (long long)bz * rows * cols;

  for (int phase = 0; phase < inners; phase += BK) {
    for (int i = 0; i < TM; ++i) {
      int row = by * BM + ty * TM + i;
      int col = phase + tx;
      if (row < rows && col < inners)
        a_shared[ty * TM + i][tx] = a[a_base + (long long)row * inners + col];
      else
        a_shared[ty * TM + i][tx] = 0.0f;
    }

    for (int i = 0; i < TN; ++i) {
      int row = phase + ty;
      int col = bx * BN + tx * TN + i;
      if (row < inners && col < cols)
        b_shared[ty][tx * TN + i] = b[b_base + (long long)row * cols + col];
      else
        b_shared[ty][tx * TN + i] = 0.0f;
    }

    __syncthreads();

    for (int k = 0; k < BK; ++k) {
      float a_reg[TM];
      float b_reg[TN];

#pragma unroll
      for (int i = 0; i < TM; ++i)
        a_reg[i] = a_shared[ty * TM + i][k];

#pragma unroll
      for (int j = 0; j < TN; ++j)
        b_reg[j] = b_shared[k][tx * TN + j];

#pragma unroll
      for (int i = 0; i < TM; ++i) {
        float av = a_reg[i];
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          c_reg[i][j] = fmaf(av, b_reg[j], c_reg[i][j]);
        }
      }
    }

    __syncthreads();
  }

  for (int i = 0; i < TM; ++i) {
    for (int j = 0; j < TN; ++j) {
      int row = by * BM + ty * TM + i;
      int col = bx * BN + tx * TN + j;
      if (row < rows && col < cols)
        c[c_base + (long long)row * cols + col] = c_reg[i][j];
    }
  }
}

__global__ void matmul_cuda_forward_nt_kernel(const float *a, const float *b, float *c, u64 batches,
                                              u64 rows, u64 inners, u64 cols) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int bz = blockIdx.z;

  __shared__ float a_shared[BM][BK + 1];
  __shared__ float b_shared[BK][BN];

  float c_reg[TM][TN] = {0.0f};

  long long a_base = (long long)bz * rows * inners;
  long long b_base = (long long)bz * inners * cols;
  long long c_base = (long long)bz * rows * cols;

  for (int phase = 0; phase < inners; phase += BK) {
    for (int i = 0; i < TM; ++i) {
      int row = by * BM + ty * TM + i;
      int col = phase + tx;
      if (row < rows && col < inners)
        a_shared[ty * TM + i][tx] = a[a_base + (long long)row * inners + col];
      else
        a_shared[ty * TM + i][tx] = 0.0f;
    }

    for (int i = 0; i < TN; ++i) {
      int row = bx * BN + tx * TN + i;
      int col = phase + ty;
      if (row < cols && col < inners)
        b_shared[ty][tx * TN + i] = b[b_base + (long long)row * inners + col];
      else
        b_shared[ty][tx * TN + i] = 0.0f;
    }

    __syncthreads();

    for (int k = 0; k < BK; ++k) {
      float a_reg[TM];
      float b_reg[TN];

#pragma unroll
      for (int i = 0; i < TM; ++i)
        a_reg[i] = a_shared[ty * TM + i][k];

#pragma unroll
      for (int j = 0; j < TN; ++j)
        b_reg[j] = b_shared[k][tx * TN + j];

#pragma unroll
      for (int i = 0; i < TM; ++i) {
        float av = a_reg[i];
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          c_reg[i][j] = fmaf(av, b_reg[j], c_reg[i][j]);
        }
      }
    }

    __syncthreads();
  }

  for (int i = 0; i < TM; ++i) {
    for (int j = 0; j < TN; ++j) {
      int row = by * BM + ty * TM + i;
      int col = bx * BN + tx * TN + j;
      if (row < rows && col < cols)
        c[c_base + (long long)row * cols + col] = c_reg[i][j];
    }
  }
}

__global__ void matmul_cuda_forward_tn_kernel(const float *a, const float *b, float *c, u64 batches,
                                              u64 rows, u64 inners, u64 cols) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int bz = blockIdx.z;

  __shared__ float a_shared[BM][BK + 1];
  __shared__ float b_shared[BK][BN];

  float c_reg[TM][TN] = {0.0f};

  long long a_base = (long long)bz * inners * rows;
  long long b_base = (long long)bz * inners * cols;
  long long c_base = (long long)bz * rows * cols;

  for (int phase = 0; phase < inners; phase += BK) {
    for (int i = 0; i < TM; ++i) {
      int row = phase + tx;
      int col = by * BM + ty * TM + i;
      if (row < inners && col < rows)
        a_shared[ty * TM + i][tx] = a[a_base + (long long)row * rows + col];
      else
        a_shared[ty * TM + i][tx] = 0.0f;
    }

    for (int i = 0; i < TN; ++i) {
      int row = phase + ty;
      int col = bx * BN + tx * TN + i;
      if (row < inners && col < cols)
        b_shared[ty][tx * TN + i] = b[b_base + (long long)row * cols + col];
      else
        b_shared[ty][tx * TN + i] = 0.0f;
    }

    __syncthreads();

    for (int k = 0; k < BK; ++k) {
      float a_reg[TM];
      float b_reg[TN];

#pragma unroll
      for (int i = 0; i < TM; ++i)
        a_reg[i] = a_shared[ty * TM + i][k];

#pragma unroll
      for (int j = 0; j < TN; ++j)
        b_reg[j] = b_shared[k][tx * TN + j];

#pragma unroll
      for (int i = 0; i < TM; ++i) {
        float av = a_reg[i];
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          c_reg[i][j] = fmaf(av, b_reg[j], c_reg[i][j]);
        }
      }
    }

    __syncthreads();
  }

  for (int i = 0; i < TM; ++i) {
    for (int j = 0; j < TN; ++j) {
      int row = by * BM + ty * TM + i;
      int col = bx * BN + tx * TN + j;
      if (row < rows && col < cols)
        c[c_base + (long long)row * cols + col] = c_reg[i][j];
    }
  }
}

extern "C" void matmul_cuda_forward(const Tensor **inputs, Tensor *output, KernelParams params) {
  const Tensor *a = inputs[0];
  const Tensor *b = inputs[1];

  u64 M = a->shape[a->ndim - 2];
  u64 K = a->shape[a->ndim - 1];
  u64 N = b->shape[b->ndim - 1];

  u64 batches = 1;
  for (u64 i = 0; i < a->ndim - 2; ++i)
    batches *= a->shape[i];

  dim3 block_dim(BN / TN, BM / TM, 1);
  dim3 grid_dim(CEIL_DIV(N, BN), CEIL_DIV(M, BM), batches);

  CudaTensorPack pa, pb;
  cuda_tensor_pack_init(&pa, a);
  cuda_tensor_pack_init(&pb, b);
  if (!pa.data || !pb.data) {
    cuda_tensor_pack_release(&pa);
    cuda_tensor_pack_release(&pb);
    return;
  }

  switch (a->dtype) {
  case FLOAT32:
    matmul_cuda_forward_contig_kernel<<<grid_dim, block_dim>>>(
        (const float *)pa.data, (const float *)pb.data, (float *)output->data, batches, M, K, N);
    break;
  default:
    fprintf(stderr, "Unsupported data type for matmul_cuda_forward\n");
    break;
  }

  cuda_tensor_pack_release(&pa);
  cuda_tensor_pack_release(&pb);
  CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" void matmul_cuda_backward(Tensor **inputs, const Tensor *output, KernelParams params) {
  Tensor *a = inputs[0];
  Tensor *b = inputs[1];
  Tensor *da = a->grad;
  Tensor *db = b->grad;
  const Tensor *dc = output->grad;

  if (!dc)
    return;

  u64 M = a->shape[a->ndim - 2];
  u64 K = a->shape[a->ndim - 1];
  u64 N = b->shape[b->ndim - 1];

  u64 batches = 1;
  for (u64 i = 0; i < a->ndim - 2; ++i)
    batches *= a->shape[i];

  CudaTensorPack pdc;
  cuda_tensor_pack_init(&pdc, dc);
  if (!pdc.data)
    return;

  dim3 opt_block(BN / TN, BM / TM, 1);

  switch (a->dtype) {
  case FLOAT32:
    if (a->requires_grad) {
      CudaTensorPack pb;
      cuda_tensor_pack_init(&pb, b);
      if (pb.data) {
        dim3 grid_dim_da(CEIL_DIV(K, BN), CEIL_DIV(M, BM), batches);
        matmul_cuda_forward_nt_kernel<<<grid_dim_da, opt_block>>>(
            (const float *)pdc.data, (const float *)pb.data, (float *)da->data, batches, M, N, K);
      }
      cuda_tensor_pack_release(&pb);
    }

    if (b->requires_grad) {
      CudaTensorPack pa;
      cuda_tensor_pack_init(&pa, a);
      if (pa.data) {
        dim3 grid_dim_db(CEIL_DIV(N, BN), CEIL_DIV(K, BM), batches);
        matmul_cuda_forward_tn_kernel<<<grid_dim_db, opt_block>>>(
            (const float *)pa.data, (const float *)pdc.data, (float *)db->data, batches, K, M, N);
      }
      cuda_tensor_pack_release(&pa);
    }
    break;
  default:
    break;
  }

  cuda_tensor_pack_release(&pdc);
}

// ─── Host-callable wrappers for NT/TN kernels (for use by fused ops) ────────

extern "C" void launch_matmul_nt_cuda(const float *a, const float *b, float *c, u64 batches,
                                      u64 rows, u64 inners, u64 cols, dim3 grid_dim,
                                      dim3 block_dim) {
  matmul_cuda_forward_nt_kernel<<<grid_dim, block_dim>>>(a, b, c, batches, rows, inners, cols);
}

extern "C" void launch_matmul_tn_cuda(const float *a, const float *b, float *c, u64 batches,
                                      u64 rows, u64 inners, u64 cols, dim3 grid_dim,
                                      dim3 block_dim) {
  matmul_cuda_forward_tn_kernel<<<grid_dim, block_dim>>>(a, b, c, batches, rows, inners, cols);
}

extern "C" void matmul_cuda_forward_direct(
    const float *a_data, const u64 *a_shape, const u64 *a_strides, u64 a_ndim,
    const float *b_data, const u64 *b_shape, const u64 *b_strides, u64 b_ndim,
    float *c_data, const u64 *c_shape, const u64 *c_strides, u64 c_ndim) {
  Tensor a = {0};
  a.data = (void*)a_data;
  a.ndim = a_ndim;
  a.dtype = FLOAT32;
  memcpy(a.shape, a_shape, a_ndim * sizeof(u64));
  memcpy(a.strides, a_strides, a_ndim * sizeof(u64));

  Tensor b = {0};
  b.data = (void*)b_data;
  b.ndim = b_ndim;
  b.dtype = FLOAT32;
  memcpy(b.shape, b_shape, b_ndim * sizeof(u64));
  memcpy(b.strides, b_strides, b_ndim * sizeof(u64));

  Tensor output = {0};
  output.data = (void*)c_data;
  output.ndim = c_ndim;
  output.dtype = FLOAT32;
  memcpy(output.shape, c_shape, c_ndim * sizeof(u64));
  memcpy(output.strides, c_strides, c_ndim * sizeof(u64));

  u64 M = a.shape[a.ndim - 2];
  u64 K = a.shape[a.ndim - 1];
  u64 N = b.shape[b.ndim - 1];

  u64 batches = 1;
  for (u64 i = 0; i < a.ndim - 2; ++i)
    batches *= a.shape[i];

  dim3 block_dim(BN / TN, BM / TM, 1);
  dim3 grid_dim(CEIL_DIV(N, BN), CEIL_DIV(M, BM), batches);

  CudaTensorPack pa, pb;
  cuda_tensor_pack_init(&pa, &a);
  cuda_tensor_pack_init(&pb, &b);
  if (!pa.data || !pb.data) {
    cuda_tensor_pack_release(&pa);
    cuda_tensor_pack_release(&pb);
    return;
  }

  matmul_cuda_forward_contig_kernel<<<grid_dim, block_dim>>>(
      (const float *)pa.data, (const float *)pb.data, (float *)output.data, batches, M, K, N);

  cuda_tensor_pack_release(&pa);
  cuda_tensor_pack_release(&pb);
  CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" void matmul_cuda_backward_direct(
    const float *a_data, const u64 *a_shape, const u64 *a_strides, u64 a_ndim,
    float *da_data, const u64 *da_strides, bool a_requires_grad,
    const float *b_data, const u64 *b_shape, const u64 *b_strides, u64 b_ndim,
    float *db_data, const u64 *db_strides, bool b_requires_grad,
    const float *dc_data, const u64 *dc_shape, const u64 *dc_strides, u64 dc_ndim) {

  Tensor a = {0};
  a.data = (void*)a_data;
  a.ndim = a_ndim;
  a.dtype = FLOAT32;
  a.requires_grad = a_requires_grad;
  memcpy(a.shape, a_shape, a_ndim * sizeof(u64));
  memcpy(a.strides, a_strides, a_ndim * sizeof(u64));

  Tensor b = {0};
  b.data = (void*)b_data;
  b.ndim = b_ndim;
  b.dtype = FLOAT32;
  b.requires_grad = b_requires_grad;
  memcpy(b.shape, b_shape, b_ndim * sizeof(u64));
  memcpy(b.strides, b_strides, b_ndim * sizeof(u64));

  Tensor output = {0};
  output.ndim = dc_ndim;
  output.dtype = FLOAT32;
  memcpy(output.shape, dc_shape, dc_ndim * sizeof(u64));
  memcpy(output.strides, dc_strides, dc_ndim * sizeof(u64));

  Tensor dc_tensor = {0};
  dc_tensor.data = (void*)dc_data;
  dc_tensor.ndim = dc_ndim;
  dc_tensor.dtype = FLOAT32;
  memcpy(dc_tensor.shape, dc_shape, dc_ndim * sizeof(u64));
  memcpy(dc_tensor.strides, dc_strides, dc_ndim * sizeof(u64));
  output.grad = &dc_tensor;

  Tensor da_tensor = {0};
  da_tensor.data = (void*)da_data;
  da_tensor.ndim = a_ndim;
  da_tensor.dtype = FLOAT32;
  memcpy(da_tensor.shape, a_shape, a_ndim * sizeof(u64));
  memcpy(da_tensor.strides, da_strides, a_ndim * sizeof(u64));
  a.grad = &da_tensor;

  Tensor db_tensor = {0};
  db_tensor.data = (void*)db_data;
  db_tensor.ndim = b_ndim;
  db_tensor.dtype = FLOAT32;
  memcpy(db_tensor.shape, b_shape, b_ndim * sizeof(u64));
  memcpy(db_tensor.strides, db_strides, b_ndim * sizeof(u64));
  b.grad = &db_tensor;

  u64 M = a.shape[a.ndim - 2];
  u64 K = a.shape[a.ndim - 1];
  u64 N = b.shape[b.ndim - 1];

  u64 batches = 1;
  for (u64 i = 0; i < a.ndim - 2; ++i)
    batches *= a.shape[i];

  CudaTensorPack pdc;
  cuda_tensor_pack_init(&pdc, &dc_tensor);
  if (!pdc.data)
    return;

  dim3 opt_block(BN / TN, BM / TM, 1);

  if (a.requires_grad) {
    CudaTensorPack pb;
    cuda_tensor_pack_init(&pb, &b);
    if (pb.data) {
      dim3 grid_dim_da(CEIL_DIV(K, BN), CEIL_DIV(M, BM), batches);
      matmul_cuda_forward_nt_kernel<<<grid_dim_da, opt_block>>>(
          (const float *)pdc.data, (const float *)pb.data, (float *)da_tensor.data, batches, M, N, K);
    }
    cuda_tensor_pack_release(&pb);
  }

  if (b.requires_grad) {
    CudaTensorPack pa;
    cuda_tensor_pack_init(&pa, &a);
    if (pa.data) {
      dim3 grid_dim_db(CEIL_DIV(N, BN), CEIL_DIV(K, BM), batches);
      matmul_cuda_forward_tn_kernel<<<grid_dim_db, opt_block>>>(
          (const float *)pa.data, (const float *)pdc.data, (float *)db_tensor.data, batches, K, M, N);
    }
    cuda_tensor_pack_release(&pa);
  }

  cuda_tensor_pack_release(&pdc);
  CUDA_CHECK(cudaDeviceSynchronize());
}
