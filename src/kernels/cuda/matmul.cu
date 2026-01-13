#include "definitions.h"
#include "kernels/transpose.h"
#include "tensor.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / b)
#define BLOCKSIZE 32

__global__ void matmul_cuda_forward_contig_kernel(const float *a,
                                                  const float *b, float *c,
                                                  u64 batches, u64 rows,
                                                  u64 inners, u64 cols) {
  const u64 tx = threadIdx.x % BLOCKSIZE;
  const u64 ty = threadIdx.x / BLOCKSIZE;

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

__global__ void matmul_cuda_forward_nt_kernel(const float *a, const float *b,
                                                float *c, u64 batches, u64 rows,
                                                u64 inners, u64 cols) {
  const u64 tx = threadIdx.x % BLOCKSIZE;
  const u64 ty = threadIdx.x / BLOCKSIZE;

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

__global__ void matmul_cuda_forward_tn_kernel(const float *a, const float *b,
                                                float *c, u64 batches, u64 rows,
                                                u64 inners, u64 cols) {
  const u64 tx = threadIdx.x % BLOCKSIZE;
  const u64 ty = threadIdx.x / BLOCKSIZE;

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

__global__ void pack_tensor_cuda_kernel(const float *src_data,
                                        const u64 *src_shape,
                                        const u64 *src_strides, u64 src_ndim,
                                        float *dst_data, u64 num_elements) {
  u64 linear_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (linear_idx < num_elements) {
    u64 coords[MAX_NDIM];
    u64 temp_idx = linear_idx;

    // linear_to_coords
    for (int i = src_ndim - 1; i >= 0; --i) {
      coords[i] = temp_idx % src_shape[i];
      temp_idx /= src_shape[i];
    }

    // get_offset
    u64 src_offset = 0;
    for (u64 i = 0; i < src_ndim; ++i) {
      src_offset += coords[i] * src_strides[i];
    }

    dst_data[linear_idx] = src_data[src_offset];
  }
}

void pack_tensor_to_contiguous_buffer_cuda(const Tensor *src, void **dst_ptr) {
  u64 num_elements = numel(src);
  u64 element_size = dtype_size(src->dtype);

  // Allocate contiguous memory on device
  cudaError_t err = cudaMalloc(dst_ptr, num_elements * element_size);
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA memory allocation failed for packed buffer: %s\n",
            cudaGetErrorString(err));
    *dst_ptr = NULL;
    return;
  }

  // Copy shape and strides to device
  u64 *d_src_shape;
  u64 *d_src_strides;
  err = cudaMalloc(&d_src_shape, src->ndim * sizeof(u64));
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA memory allocation failed for d_src_shape: %s\n",
            cudaGetErrorString(err));
    cudaFree(*dst_ptr);
    *dst_ptr = NULL;
    return;
  }
  err = cudaMalloc(&d_src_strides, src->ndim * sizeof(u64));
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA memory allocation failed for d_src_strides: %s\n",
            cudaGetErrorString(err));
    cudaFree(d_src_shape);
    cudaFree(*dst_ptr);
    *dst_ptr = NULL;
    return;
  }

  err = cudaMemcpy(d_src_shape, src->shape, src->ndim * sizeof(u64),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA memcpy failed for d_src_shape: %s\n",
            cudaGetErrorString(err));
    cudaFree(d_src_shape);
    cudaFree(d_src_strides);
    *dst_ptr = NULL;
    return;
  }
  err = cudaMemcpy(d_src_strides, src->strides, src->ndim * sizeof(u64),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA memcpy failed for d_src_strides: %s\n",
            cudaGetErrorString(err));
    cudaFree(d_src_shape);
    cudaFree(d_src_strides);
    *dst_ptr = NULL;
    return;
  }

  // Launch kernel
  u64 threads_per_block = 256;
  u64 num_blocks = CEIL_DIV(num_elements, threads_per_block);

  pack_tensor_cuda_kernel<<<num_blocks, threads_per_block>>>(
      (const float *)src->data, d_src_shape, d_src_strides, src->ndim,
      (float *)*dst_ptr, num_elements);

  cudaFree(d_src_shape);
  cudaFree(d_src_strides);
}

extern "C" void matmul_cuda_forward(const Tensor **inputs, Tensor *output,
                                    ...) {
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

  dim3 block_dim(BLOCKSIZE, BLOCKSIZE, 1);
  dim3 grid_dim(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE), batches);

  void *a_data_ptr = a->data;
  void *b_data_ptr = b->data;

  void *a_packed_data = NULL;
  void *b_packed_data = NULL;

  if (!is_contiguous(a)) {
    pack_tensor_to_contiguous_buffer_cuda(a, &a_packed_data);
    if (a_packed_data == NULL) {
      fprintf(stderr, "Failed to pack tensor A in matmul_cuda_forward\n");
      return;
    }
    a_data_ptr = a_packed_data;
  }

  if (!is_contiguous(b)) {
    pack_tensor_to_contiguous_buffer_cuda(b, &b_packed_data);
    if (b_packed_data == NULL) {
      fprintf(stderr, "Failed to pack tensor B in matmul_cuda_forward\n");
      if (a_packed_data)
        cudaFree(a_packed_data);
      return;
    }
    b_data_ptr = b_packed_data;
  }

  switch (a->dtype) {
  case FLOAT32:
    matmul_cuda_forward_contig_kernel<<<grid_dim, block_dim>>>(
        (const float *)a_data_ptr, (const float *)b_data_ptr,
        (float *)output->data, batches, M, K, N);
    break;
  default:
    fprintf(stderr, "Unsupported data type for matmul_cuda_forward\n");
    break;
  }

  if (a_packed_data)
    cudaFree(a_packed_data);
  if (b_packed_data)
    cudaFree(b_packed_data);
}

extern "C" void matmul_cuda_backward(Tensor **inputs, const Tensor *output,
                                     ...) {
  Tensor *a = inputs[0];
  Tensor *b = inputs[1];
  Tensor *da = a->grad;
  Tensor *db = b->grad;
  const Tensor *dc = output->grad;

  u64 a_ndim = a->ndim;
  u64 b_ndim = b->ndim;

  u64 M = a->shape[a_ndim - 2];
  u64 K = a->shape[a_ndim - 1];
  u64 N = b->shape[b_ndim - 1];

  u64 batches = 1;
  for (u64 i = 0; i < a_ndim - 2; ++i) {
    batches *= a->shape[i];
  }

  // Pointers for potentially packed data on device
  void *dc_data_ptr = dc->data;
  void *dc_packed_data = NULL;

  if (!is_contiguous(dc)) {
    pack_tensor_to_contiguous_buffer_cuda(dc, &dc_packed_data);
    if (dc_packed_data == NULL) {
      fprintf(stderr, "Failed to pack tensor DC in matmul_cuda_backward\n");
      return;
    }
    dc_data_ptr = dc_packed_data;
  }

  switch (a->dtype) {
  case FLOAT32:
    if (a->requires_grad) {
      // NOTE: da = dc @ B.T
      // NT Kernel: X=dc, Y=b, Z=da.
      // dc: (batches, M, N)
      // b: (batches, K, N)
      // da: (batches, M, K)
      
      // Define grid and block dimensions for this kernel call
      dim3 block_dim_da(BLOCKSIZE, BLOCKSIZE, 1);
      // NT Kernel args: (batches, rows, inners, cols) -> (batches, M, N, K)
      // rows=M, inners=N (common dim), cols=K (output cols)
      dim3 grid_dim_da(CEIL_DIV(K, BLOCKSIZE), CEIL_DIV(M, BLOCKSIZE), batches);
      
      // Need to pass packed data if available, or raw data
      const float *dc_ptr = (const float *)dc_data_ptr;
      const float *b_ptr = (const float *)b->data;
      void *b_packed = NULL;
      
      if (!is_contiguous(b)) {
        pack_tensor_to_contiguous_buffer_cuda(b, &b_packed);
        if (b_packed == NULL) {
            fprintf(stderr, "Failed to pack B in matmul_cuda_backward\n");
            if (dc_packed_data) cudaFree(dc_packed_data);
            return;
        }
        b_ptr = (const float *)b_packed;
      }

      matmul_cuda_forward_nt_kernel<<<grid_dim_da, block_dim_da>>>(
          dc_ptr, b_ptr,
          (float *)da->data, batches, M, N, K);

      if (b_packed) cudaFree(b_packed);
    }

    if (b->requires_grad) {
      // NOTE: db = A.T @ dc
      // TN Kernel: X=a, Y=dc, Z=db.
      // a: (batches, M, K)
      // dc: (batches, M, N)
      // db: (batches, K, N)
      
      // TN Kernel args: (batches, rows, inners, cols) -> (batches, K, M, N)
      // rows=K, inners=M (common dim), cols=N (output cols)
      dim3 block_dim_db(BLOCKSIZE, BLOCKSIZE, 1);
      dim3 grid_dim_db(CEIL_DIV(N, BLOCKSIZE), CEIL_DIV(K, BLOCKSIZE), batches);
      
      const float *a_ptr = (const float *)a->data;
      const float *dc_ptr = (const float *)dc_data_ptr;
      void *a_packed = NULL;
      
      if (!is_contiguous(a)) {
        pack_tensor_to_contiguous_buffer_cuda(a, &a_packed);
        if (a_packed == NULL) {
            fprintf(stderr, "Failed to pack A in matmul_cuda_backward\n");
            if (dc_packed_data) cudaFree(dc_packed_data);
            return;
        }
        a_ptr = (const float *)a_packed;
      }
      
      matmul_cuda_forward_tn_kernel<<<grid_dim_db, block_dim_db>>>(
          a_ptr, dc_ptr,
          (float *)db->data, batches, K, M, N);
          
      if (a_packed) cudaFree(a_packed);
    }
    break;
  default:
    break;
  }

  if (dc_packed_data)
    cudaFree(dc_packed_data);
}
