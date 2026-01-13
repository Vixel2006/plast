#include "kernels/matmul.h"
#include "definitions.h"
#include "kernels/cpu_utils.h"
#include "kernels/transpose.h"
#include "tensor.h"
#include <omp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TILE_SIZE 32

void pack_tensor_to_contiguous_buffer(const Tensor *src, void *dst) {
  u64 num_elements = numel(src);
  u64 element_size = dtype_size(src->dtype);

  if (is_contiguous(src)) {
    memcpy(dst, src->data, num_elements * element_size);
    return;
  }

  u64 *coords = (u64 *)malloc(src->ndim * sizeof(u64));
  if (!coords) {
    fprintf(stderr, "Memory allocation failed for coords in "
                    "pack_tensor_to_contiguous_buffer\n");
    return;
  }

  for (u64 i = 0; i < num_elements; ++i) {
    linear_to_coords(i, src->shape, src->ndim, coords);
    u64 src_offset = get_offset(coords, src->strides, src->ndim);
    memcpy((u8 *)dst + i * element_size, (u8 *)src->data + src_offset,
           element_size);
  }

  free(coords);
}

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

void matmul_cpu_forward(const Tensor **inputs, Tensor *output, ...) {
  const Tensor *a = inputs[0];
  const Tensor *b = inputs[1];

  u64 a_ndim = a->ndim;
  u64 b_ndim = b->ndim;

  u64 M = a->shape[a_ndim - 2];
  u64 K = a->shape[a_ndim - 1];
  u64 N = b->shape[b_ndim - 1];

  u64 batches = 1;
  for (u64 i = 0; i < a_ndim - 2; ++i) {
    batches *= a->shape[i];
  }

  if (is_contiguous(a) && is_contiguous(b)) {
    switch (a->dtype) {
    case FLOAT32:
      matmul_cpu_forward_float_contig_kernel(
          (const float *)a->data, (const float *)b->data, (float *)output->data,
          batches, M, K, N);
      break;
    default:
      fprintf(stderr, "Unsupported data type for matmul_cpu_forward\n");
      break;
    }
  } else {
    void *a_data_ptr = (void *)a->data;
    void *b_data_ptr = (void *)b->data;

    void *a_packed_data = NULL;
    void *b_packed_data = NULL;

    u64 element_size = dtype_size(a->dtype);

    if (!is_contiguous(a)) {
      a_packed_data = malloc(numel(a) * element_size);
      if (!a_packed_data) {
        fprintf(stderr, "Memory allocation failed for packed_a_data in "
                        "matmul_cpu_forward\n");
        return;
      }
      pack_tensor_to_contiguous_buffer(a, a_packed_data);
      a_data_ptr = a_packed_data;
    }

    if (!is_contiguous(b)) {
      b_packed_data = malloc(numel(b) * element_size);
      if (!b_packed_data) {
        fprintf(stderr, "Memory allocation failed for packed_b_data in "
                        "matmul_cpu_forward\n");
        free(a_packed_data);
        return;
      }
      pack_tensor_to_contiguous_buffer(b, b_packed_data);
      b_data_ptr = b_packed_data;
    }

    switch (a->dtype) {
    case FLOAT32:
      matmul_cpu_forward_float_contig_kernel(
          (const float *)a_data_ptr, (const float *)b_data_ptr,
          (float *)output->data, batches, M, K, N);
      break;
    default:
      fprintf(stderr, "Unsupported data type for matmul_cpu_forward\n");
      break;
    }

    free(a_packed_data);
    free(b_packed_data);
  }
}

void matmul_cpu_backward(Tensor **inputs, const Tensor *output, ...) {
  Tensor *a = inputs[0];
  Tensor *b = inputs[1];
  Tensor *da = a->grad;
  Tensor *db = b->grad;
  const Tensor *dc = output->grad;

  if (dc == NULL) {
    return;
  }

  u64 M = a->shape[a->ndim - 2];
  u64 K = a->shape[a->ndim - 1];
  u64 N = b->shape[b->ndim - 1];

  u64 batches = 1;
  for (u64 i = 0; i < a->ndim - 2; ++i) {
    batches *= a->shape[i];
  }

  void *dc_data_ptr = dc->data;
  void *dc_packed_data = NULL;
  u64 element_size = dtype_size(a->dtype);

  if (!is_contiguous(dc)) {
    dc_packed_data = malloc(numel(dc) * element_size);
    if (!dc_packed_data) {
      fprintf(stderr, "Memory allocation failed for dc_packed_data in "
                      "matmul_cpu_backward\n");
      return;
    }
    pack_tensor_to_contiguous_buffer(dc, dc_packed_data);
    dc_data_ptr = dc_packed_data;
  }

  if (a->requires_grad) {
    void *b_data_ptr = b->data;
    void *b_packed_data = NULL;
    if (!is_contiguous(b)) {
      b_packed_data = malloc(numel(b) * element_size);
      if (!b_packed_data) {
        fprintf(stderr, "Memory allocation failed for b_packed_data in "
                        "matmul_cpu_backward\n");
        free(dc_packed_data);
        return;
      }
      pack_tensor_to_contiguous_buffer(b, b_packed_data);
      b_data_ptr = b_packed_data;
    }

    matmul_cpu_forward_float_nt_kernel((const float *)dc_data_ptr,
                                       (const float *)b_data_ptr,
                                       (float *)da->data, batches, M, N, K);
    free(b_packed_data);
  }

  if (b->requires_grad) {
    void *a_data_ptr = a->data;
    void *a_packed_data = NULL;
    if (!is_contiguous(a)) {
      a_packed_data = malloc(numel(a) * element_size);
      if (!a_packed_data) {
        fprintf(stderr, "Memory allocation failed for a_packed_data in "
                        "matmul_cpu_backward\n");
        free(dc_packed_data);
        return;
      }
      pack_tensor_to_contiguous_buffer(a, a_packed_data);
      a_data_ptr = a_packed_data;
    }

    matmul_cpu_forward_float_tn_kernel((const float *)a_data_ptr,
                                       (const float *)dc_data_ptr,
                                       (float *)db->data, batches, K, M, N);
    free(a_packed_data);
  }

  free(dc_packed_data);
}
