#include "cuda_check.h"
#include "definitions.h"
#include "kernels/pack.h"
#include "kernels/transpose.h"
#include "tensor.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / b)
#define BLOCKSIZE 32

__global__ void matmul_cuda_forward_contig_kernel(const float *a, const float *b, float *c,
                                                  u64 batches, u64 rows, u64 inners, u64 cols) {
  const u64 tx = threadIdx.x;
  const u64 ty = threadIdx.y;

  const u64 col = blockIdx.x * BLOCKSIZE + tx;
  const u64 row = blockIdx.y * BLOCKSIZE + ty;
  const u64 batch = blockIdx.z;

  __shared__ float a_tiled[BLOCKSIZE][BLOCKSIZE];
  __shared__ float b_tiled[BLOCKSIZE][BLOCKSIZE];

  float accum = 0.0f;

  for (u64 phase = 0; phase < inners; phase += BLOCKSIZE) {
    if (row < rows && (phase + tx) < inners)
      a_tiled[ty][tx] = a[batch * rows * inners + row * inners + (phase + tx)];
    else
      a_tiled[ty][tx] = 0.0f;
    if ((phase + ty) < inners && col < cols)
      b_tiled[ty][tx] = b[batch * inners * cols + (phase + ty) * cols + col];
    else
      b_tiled[ty][tx] = 0.0f;

    __syncthreads();

    for (u64 k = 0; k < BLOCKSIZE; ++k) {
      accum += a_tiled[ty][k] * b_tiled[k][tx];
    }

    __syncthreads();
  }

  if (row < rows && col < cols) {
    c[batch * rows * cols + row * cols + col] = accum;
  }
}

__global__ void matmul_cuda_forward_nt_kernel(const float *a, const float *b, float *c, u64 batches,
                                              u64 rows, u64 inners, u64 cols) {
  const u64 tx = threadIdx.x;
  const u64 ty = threadIdx.y;

  const u64 col = blockIdx.x * BLOCKSIZE + tx;
  const u64 row = blockIdx.y * BLOCKSIZE + ty;
  const u64 batch = blockIdx.z;

  __shared__ float a_tiled[BLOCKSIZE][BLOCKSIZE];
  __shared__ float b_tiled[BLOCKSIZE][BLOCKSIZE];

  float accum = 0.0f;

  for (u64 phase = 0; phase < inners; phase += BLOCKSIZE) {
    if (row < rows && (phase + tx) < inners)
      a_tiled[ty][tx] = a[batch * rows * inners + row * inners + (phase + tx)];
    else
      a_tiled[ty][tx] = 0.0f;

    u64 B_row = blockIdx.x * BLOCKSIZE + tx;
    u64 B_col = phase + ty;

    if (B_row < cols && B_col < inners)
      b_tiled[ty][tx] = b[batch * cols * inners + B_row * inners + B_col];
    else
      b_tiled[ty][tx] = 0.0f;

    __syncthreads();

    for (u64 k = 0; k < BLOCKSIZE; ++k) {
      accum += a_tiled[ty][k] * b_tiled[k][tx];
    }

    __syncthreads();
  }

  if (row < rows && col < cols) {
    c[batch * rows * cols + row * cols + col] = accum;
  }
}

__global__ void matmul_cuda_forward_tn_kernel(const float *a, const float *b, float *c, u64 batches,
                                              u64 rows, u64 inners, u64 cols) {
  const u64 tx = threadIdx.x;
  const u64 ty = threadIdx.y;

  const u64 col = blockIdx.x * BLOCKSIZE + tx;
  const u64 row = blockIdx.y * BLOCKSIZE + ty;
  const u64 batch = blockIdx.z;

  __shared__ float a_tiled[BLOCKSIZE][BLOCKSIZE];
  __shared__ float b_tiled[BLOCKSIZE][BLOCKSIZE];

  float accum = 0.0f;

  for (u64 phase = 0; phase < inners; phase += BLOCKSIZE) {
    u64 load_a_row = phase + tx;
    u64 load_a_col = blockIdx.y * BLOCKSIZE + ty;

    if (load_a_row < inners && load_a_col < rows)
      a_tiled[ty][tx] = a[batch * inners * rows + load_a_row * rows + load_a_col];
    else
      a_tiled[ty][tx] = 0.0f;

    u64 load_b_row = phase + ty;
    u64 load_b_col = blockIdx.x * BLOCKSIZE + tx;

    if (load_b_row < inners && load_b_col < cols)
      b_tiled[ty][tx] = b[batch * inners * cols + load_b_row * cols + load_b_col];
    else
      b_tiled[ty][tx] = 0.0f;

    __syncthreads();

    for (u64 k = 0; k < BLOCKSIZE; ++k) {
      accum += a_tiled[ty][k] * b_tiled[k][tx];
    }

    __syncthreads();
  }

  if (row < rows && col < cols) {
    c[batch * rows * cols + row * cols + col] = accum;
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

  dim3 block_dim(BLOCKSIZE, BLOCKSIZE, 1);
  dim3 grid_dim(CEIL_DIV(N, BLOCKSIZE), CEIL_DIV(M, BLOCKSIZE), batches);

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

  switch (a->dtype) {
  case FLOAT32:
    if (a->requires_grad) {
      CudaTensorPack pb;
      cuda_tensor_pack_init(&pb, b);
      if (pb.data) {
        dim3 block_dim_da(BLOCKSIZE, BLOCKSIZE, 1);
        dim3 grid_dim_da(CEIL_DIV(K, BLOCKSIZE), CEIL_DIV(M, BLOCKSIZE), batches);
        matmul_cuda_forward_nt_kernel<<<grid_dim_da, block_dim_da>>>(
            (const float *)pdc.data, (const float *)pb.data, (float *)da->data, batches, M, N, K);
      }
      cuda_tensor_pack_release(&pb);
    }

    if (b->requires_grad) {
      CudaTensorPack pa;
      cuda_tensor_pack_init(&pa, a);
      if (pa.data) {
        dim3 block_dim_db(BLOCKSIZE, BLOCKSIZE, 1);
        dim3 grid_dim_db(CEIL_DIV(N, BLOCKSIZE), CEIL_DIV(K, BLOCKSIZE), batches);
        matmul_cuda_forward_tn_kernel<<<grid_dim_db, block_dim_db>>>(
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
