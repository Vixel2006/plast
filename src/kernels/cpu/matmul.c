#include "kernels/matmul.h"
#include "kernels/pack.h"
#include "kernels/cpu_utils.h"
#include "kernels/transpose.h"
#include "tensor.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TILE_SIZE 32

void matmul_cpu_forward_float_contig_kernel(const float *a, const float *b,
                                            float *c, u64 batches, u64 rows,
                                            u64 inners, u64 cols) {
#pragma omp parallel for collapse(2) num_threads(8)
  for (u64 batch = 0; batch < batches; ++batch) {
    for (u64 row_tile = 0; row_tile < rows; row_tile += TILE_SIZE) {
      u64 row_tile_end = MIN(rows, row_tile + TILE_SIZE);
      for (u64 inner_tile = 0; inner_tile < inners; inner_tile += TILE_SIZE) {
        u64 inner_tile_end = MIN(inners, inner_tile + TILE_SIZE);
        for (u64 col_tile = 0; col_tile < cols; col_tile += TILE_SIZE) {
          u64 col_tile_end = MIN(cols, col_tile + TILE_SIZE);
          for (u64 row = row_tile; row < row_tile_end; ++row) {
            for (u64 inner = inner_tile; inner < inner_tile_end; ++inner) {
              for (u64 col = col_tile; col < col_tile_end; ++col) {
                c[batch * rows * cols + row * cols + col] += 
                    a[batch * rows * inners + row * inners + inner] *
                    b[batch * inners * cols + inner * cols + col];
              }
            }
          }
        }
      }
    }
  }
}

void matmul_cpu_forward_float_nt_kernel(const float *a, const float *b,
                                            float *c, u64 batches, u64 rows,
                                            u64 inners, u64 cols) {
#pragma omp parallel for collapse(2) num_threads(8)
  for (u64 batch = 0; batch < batches; ++batch) {
    for (u64 row_tile = 0; row_tile < rows; row_tile += TILE_SIZE) {
      u64 row_tile_end = MIN(rows, row_tile + TILE_SIZE);
      for (u64 col_tile = 0; col_tile < cols; col_tile += TILE_SIZE) {
        u64 col_tile_end = MIN(cols, col_tile + TILE_SIZE);
        for (u64 row = row_tile; row < row_tile_end; ++row) {
          for (u64 col = col_tile; col < col_tile_end; ++col) {
            float sum = 0.0f;
            for (u64 inner = 0; inner < inners; ++inner) {
              sum += a[batch * rows * inners + row * inners + inner] *
                     b[batch * cols * inners + col * inners + inner];
            }
            c[batch * rows * cols + row * cols + col] += sum;
          }
        }
      }
    }
  }
}

void matmul_cpu_forward_float_tn_kernel(const float *a, const float *b,
                                            float *c, u64 batches, u64 rows,
                                            u64 inners, u64 cols) {
#pragma omp parallel for collapse(2) num_threads(8)
  for (u64 batch = 0; batch < batches; ++batch) {
    for (u64 row_tile = 0; row_tile < rows; row_tile += TILE_SIZE) {
      u64 row_tile_end = MIN(rows, row_tile + TILE_SIZE);
      for (u64 col_tile = 0; col_tile < cols; col_tile += TILE_SIZE) {
        u64 col_tile_end = MIN(cols, col_tile + TILE_SIZE);
        for (u64 row = row_tile; row < row_tile_end; ++row) {
          for (u64 col = col_tile; col < col_tile_end; ++col) {
            float sum = 0.0f;
            for (u64 inner = 0; inner < inners; ++inner) {
              sum += a[batch * inners * rows + inner * rows + row] *
                     b[batch * inners * cols + inner * cols + col];
            }
            c[batch * rows * cols + row * cols + col] += sum;
          }
        }
      }
    }
  }
}

void matmul_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params) {
  const Tensor *a = inputs[0];
  const Tensor *b = inputs[1];

  u64 M = a->shape[a->ndim - 2];
  u64 K = a->shape[a->ndim - 1];
  u64 N = b->shape[b->ndim - 1];

  u64 batches = 1;
  for (u64 i = 0; i < a->ndim - 2; ++i)
    batches *= a->shape[i];

  TensorPack pa, pb;
  tensor_pack_init(&pa, a);
  tensor_pack_init(&pb, b);
  if (!pa.data || !pb.data) {
    tensor_pack_release(&pa);
    tensor_pack_release(&pb);
    return;
  }

  switch (a->dtype) {
  case FLOAT32:
    matmul_cpu_forward_float_contig_kernel(
        (const float *)pa.data, (const float *)pb.data,
        (float *)output->data, batches, M, K, N);
    break;
  default:
    fprintf(stderr, "Unsupported data type for matmul_cpu_forward\n");
    break;
  }

  tensor_pack_release(&pa);
  tensor_pack_release(&pb);
}

void matmul_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params) {
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

  TensorPack pdc;
  tensor_pack_init(&pdc, dc);

  if (a->requires_grad) {
    TensorPack pb;
    tensor_pack_init(&pb, b);
    if (pb.data) {
      matmul_cpu_forward_float_nt_kernel((const float *)pdc.data,
                                         (const float *)pb.data,
                                         (float *)da->data, batches, M, N, K);
    }
    tensor_pack_release(&pb);
  }

  if (b->requires_grad) {
    TensorPack pa;
    tensor_pack_init(&pa, a);
    if (pa.data) {
      matmul_cpu_forward_float_tn_kernel((const float *)pa.data,
                                         (const float *)pdc.data,
                                         (float *)db->data, batches, K, M, N);
    }
    tensor_pack_release(&pa);
  }

  tensor_pack_release(&pdc);
}
