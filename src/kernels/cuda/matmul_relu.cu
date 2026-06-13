#include "cuda_check.h"
#include "core/definitions.h"
#include "kernels/pack.h"
#include "kernels/matmul.h"
#include "kernels/ops/shape.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

#define TM 8
#define TN 8
#define BK 16
#define BM 128
#define BN 128

__global__ void matmul_relu_cuda_forward_contig_kernel(const float *a, const float *b, float *c,
                                                       u64 batches, u64 rows, u64 inners, u64 cols,
                                                       float alpha) {
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

  for (int phase = 0; phase < (int)inners; phase += BK) {
    for (int i = 0; i < TM; ++i) {
      int row = by * BM + ty * TM + i;
      int col = phase + tx;
      if (row < (int)rows && col < (int)inners)
        a_shared[ty * TM + i][tx] = a[a_base + (long long)row * inners + col];
      else
        a_shared[ty * TM + i][tx] = 0.0f;
    }

    for (int i = 0; i < TN; ++i) {
      int row = phase + ty;
      int col = bx * BN + tx * TN + i;
      if (row < (int)inners && col < (int)cols)
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
      if (row < (int)rows && col < (int)cols) {
        float val = c_reg[i][j];
        c[c_base + (long long)row * cols + col] = val > 0.0f ? val : val * alpha;
      }
    }
  }
}

extern "C" void matmul_relu_cuda_forward(const Tensor **inputs, Tensor *output,
                                         KernelParams params) {
  const Tensor *a = inputs[0];
  const Tensor *b = inputs[1];
  float alpha = params.fval;

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
    matmul_relu_cuda_forward_contig_kernel<<<grid_dim, block_dim>>>(
        (const float *)pa.data, (const float *)pb.data, (float *)output->data, batches, M, K, N,
        alpha);
    break;
  default:
    fprintf(stderr, "Unsupported data type for matmul_relu_cuda_forward\n");
    break;
  }

  cuda_tensor_pack_release(&pa);
  cuda_tensor_pack_release(&pb);
  CUDA_CHECK(cudaDeviceSynchronize());
}

// ─── Relu gradient modulation (used by all fused backward passes) ──────────

__global__ void relu_grad_modulate_kernel(const float *dout, const float *out_data, float *dc,
                                          u64 num_elements, float alpha) {
  u64 idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float o = out_data[idx];
    dc[idx] = dout[idx] * (o > 0.0f ? 1.0f : alpha);
  }
}

extern "C" void launch_relu_grad_modulate_cuda(const float *dout, const float *out_data, float *dc,
                                               u64 num_elements, float alpha, int grid_size,
                                               int block_size) {
  relu_grad_modulate_kernel<<<grid_size, block_size>>>(dout, out_data, dc, num_elements, alpha);
}

// ─── Backward ───────────────────────────────────────────────────────────────

extern "C" void matmul_relu_cuda_backward(Tensor **inputs, const Tensor *output,
                                          KernelParams params) {
  Tensor *a = inputs[0];
  Tensor *b = inputs[1];
  Tensor *da = a->grad;
  Tensor *db = b->grad;
  const Tensor *dc = output->grad;
  float alpha = params.fval;

  if (!dc)
    return;

  u64 M = a->shape[a->ndim - 2];
  u64 K = a->shape[a->ndim - 1];
  u64 N = b->shape[b->ndim - 1];

  u64 batches = 1;
  for (u64 i = 0; i < a->ndim - 2; ++i)
    batches *= a->shape[i];

  u64 num_elements = numel(output);

  float *dc_mod = NULL;
  if (cudaMalloc(&dc_mod, num_elements * sizeof(float)) != cudaSuccess)
    return;

  int block_size = 256;
  int grid_size = CEIL_DIV(num_elements, (u64)block_size);

  launch_relu_grad_modulate_cuda((const float *)dc->data, (const float *)output->data, dc_mod,
                                 num_elements, alpha, grid_size, block_size);

  dim3 opt_block(BN / TN, BM / TM, 1);

  switch (a->dtype) {
  case FLOAT32:
    if (a->requires_grad) {
      CudaTensorPack pb;
      cuda_tensor_pack_init(&pb, b);
      if (pb.data) {
        dim3 grid_dim_da(CEIL_DIV(K, BN), CEIL_DIV(M, BM), batches);
        launch_matmul_nt_cuda(dc_mod, (const float *)pb.data, (float *)da->data, batches, M, N, K,
                              grid_dim_da, opt_block);
      }
      cuda_tensor_pack_release(&pb);
    }

    if (b->requires_grad) {
      CudaTensorPack pa;
      cuda_tensor_pack_init(&pa, a);
      if (pa.data) {
        dim3 grid_dim_db(CEIL_DIV(N, BN), CEIL_DIV(K, BM), batches);
        launch_matmul_tn_cuda((const float *)pa.data, dc_mod, (float *)db->data, batches, K, M, N,
                              grid_dim_db, opt_block);
      }
      cuda_tensor_pack_release(&pa);
    }
    break;
  default:
    break;
  }

  cudaFree(dc_mod);
}
