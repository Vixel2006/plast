#include "kernels/matmul.h"
#include "kernels/cpu_utils.h"
#include "kernels/pack.h"
#include "kernels/ops/shape.h"
#include "core/tensor.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TILE_SIZE 32

void matmul_cpu_forward_float_contig_kernel(const float *a, const float *b, float *c, u64 batches,
                                            u64 rows, u64 inners, u64 cols) {
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

void matmul_cpu_forward_float_nt_kernel(const float *a, const float *b, float *c, u64 batches,
                                        u64 rows, u64 inners, u64 cols) {
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

void matmul_cpu_forward_float_tn_kernel(const float *a, const float *b, float *c, u64 batches,
                                        u64 rows, u64 inners, u64 cols) {
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
    matmul_cpu_forward_float_contig_kernel((const float *)pa.data, (const float *)pb.data,
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
      matmul_cpu_forward_float_nt_kernel((const float *)pdc.data, (const float *)pb.data,
                                         (float *)da->data, batches, M, N, K);
    }
    tensor_pack_release(&pb);
  }

  if (b->requires_grad) {
    TensorPack pa;
    tensor_pack_init(&pa, a);
    if (pa.data) {
      matmul_cpu_forward_float_tn_kernel((const float *)pa.data, (const float *)pdc.data,
                                         (float *)db->data, batches, K, M, N);
    }
    tensor_pack_release(&pa);
  }

  tensor_pack_release(&pdc);
}

void matmul_cpu_forward_direct(
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

  u64 M = a.shape[a.ndim - 2];
  u64 K = a.shape[a.ndim - 1];
  u64 N = b.shape[b.ndim - 1];

  u64 batches = 1;
  for (u64 i = 0; i < a.ndim - 2; ++i)
    batches *= a.shape[i];

  TensorPack pa, pb;
  tensor_pack_init(&pa, &a);
  tensor_pack_init(&pb, &b);

  matmul_cpu_forward_float_contig_kernel((const float *)pa.data, (const float *)pb.data, c_data, batches, M, K, N);

  tensor_pack_release(&pa);
  tensor_pack_release(&pb);
}

void matmul_cpu_backward_direct(
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

  TensorPack pdc;
  tensor_pack_init(&pdc, &dc_tensor);

  if (a.requires_grad) {
    TensorPack pb;
    tensor_pack_init(&pb, &b);
    if (pb.data) {
      matmul_cpu_forward_float_nt_kernel((const float *)pdc.data, (const float *)pb.data,
                                         (float *)da_tensor.data, batches, M, N, K);
    }
    tensor_pack_release(&pb);
  }

  if (b.requires_grad) {
    TensorPack pa;
    tensor_pack_init(&pa, &a);
    if (pa.data) {
      matmul_cpu_forward_float_tn_kernel((const float *)pa.data, (const float *)pdc.data,
                                         (float *)db_tensor.data, batches, K, M, N);
    }
    tensor_pack_release(&pa);
  }

  tensor_pack_release(&pdc);
}
