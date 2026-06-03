#include "cuda_check.h"
#include "kernels/cuda/cuda_utils.cuh"
#include "kernels/pack.h"

template <typename T>
__global__ void pack_tensor_cuda_kernel(const T *src_data, const u64 *src_shape,
                                        const u64 *src_strides, u64 src_ndim, T *dst_data,
                                        u64 num_elements) {
  u64 idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_elements)
    return;

  u64 coords[MAX_NDIM];
  u64 tmp = idx;
  for (int i = (int)src_ndim - 1; i >= 0; --i) {
    coords[i] = tmp % src_shape[i];
    tmp /= src_shape[i];
  }
  u64 offset = 0;
  for (u64 i = 0; i < src_ndim; ++i)
    offset += coords[i] * src_strides[i];

  dst_data[idx] = src_data[offset];
}

template __global__ void pack_tensor_cuda_kernel<float>(const float *, const u64 *, const u64 *,
                                                        u64, float *, u64);

void cuda_tensor_pack_init(CudaTensorPack *p, const Tensor *t) {
  p->data = NULL;
  p->_d_buf = NULL;

  if (is_contiguous(t)) {
    p->data = t->data;
    return;
  }

  u64 num_elements = numel(t);
  u64 element_size = dtype_size(t->dtype);
  u64 *d_shape = NULL;
  u64 *d_strides = NULL;
  bool shape_ok = false;
  bool strides_ok = false;

  if (cudaMalloc(&p->_d_buf, num_elements * element_size) != cudaSuccess)
    goto fail;
  p->data = p->_d_buf;

  if (cudaMalloc(&d_shape, t->ndim * sizeof(u64)) != cudaSuccess)
    goto cleanup_buf;
  shape_ok = true;

  if (cudaMalloc(&d_strides, t->ndim * sizeof(u64)) != cudaSuccess)
    goto cleanup_shape;
  strides_ok = true;

  if (cudaMemcpy(d_shape, t->shape, t->ndim * sizeof(u64), cudaMemcpyHostToDevice) != cudaSuccess)
    goto cleanup_strides;
  if (cudaMemcpy(d_strides, t->strides, t->ndim * sizeof(u64), cudaMemcpyHostToDevice) !=
      cudaSuccess)
    goto cleanup_strides;

  {
    u64 threads_per_block = 256;
    u64 num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    pack_tensor_cuda_kernel<<<num_blocks, threads_per_block>>>(
        (const float *)t->data, d_shape, d_strides, t->ndim, (float *)p->_d_buf, num_elements);
  }

  cudaFree(d_strides);
  cudaFree(d_shape);
  return;

cleanup_strides:
  if (strides_ok)
    cudaFree(d_strides);
cleanup_shape:
  if (shape_ok)
    cudaFree(d_shape);
cleanup_buf:
  if (p->_d_buf) {
    cudaFree(p->_d_buf);
    p->_d_buf = NULL;
  }
fail:
  p->data = NULL;
}

void cuda_tensor_pack_release(CudaTensorPack *p) {
  if (p->_d_buf) {
    cudaFree(p->_d_buf);
    p->_d_buf = NULL;
  }
  p->data = NULL;
}
