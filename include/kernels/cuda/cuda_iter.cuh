#pragma once

#include "core/definitions.h"
#include "core/tensor.h"

// CUDA device-side TensorIter (lightweight — stores only what GPU kernels need)
typedef struct {
  float *data;
  u64 numel;
  bool contiguous;
  u64 shape[MAX_NDIM];
  u64 strides[MAX_NDIM];
  u64 ndim;
  // For broadcast access: precomputed dim offset
  int dim_offset;
} CudaTensorIter;

__device__ __forceinline__ u64 cuda_iter_get_offset(const u64 *coords, const u64 *strides,
                                                    u64 ndim) {
  u64 offset = 0;
  for (u64 i = 0; i < ndim; ++i)
    offset += coords[i] * strides[i];
  return offset;
}

__device__ __forceinline__ void cuda_iter_linear_to_coords(u64 idx, const u64 *shape, u64 ndim,
                                                           u64 *coords) {
  for (u64 i = ndim; i-- > 0;) {
    coords[i] = idx % shape[i];
    idx /= shape[i];
  }
}

// Broadcast-aware offset: for a tensor with fewer dims, aligns to the right
__device__ __forceinline__ u64 cuda_iter_get_offset_broadcast(const u64 *coords, u64 ndim,
                                                              const u64 *t_strides,
                                                              const u64 *t_shape, u64 t_ndim) {
  u64 offset = 0;
  int dim_offset = (int)ndim - (int)t_ndim;
  for (u64 i = 0; i < t_ndim; ++i) {
    u64 coord = coords[i + dim_offset];
    if (t_shape[i] == 1)
      coord = 0;
    offset += coord * t_strides[i];
  }
  return offset;
}

// Initialize a CudaTensorIter from a Tensor (callable from host)
void cuda_iter_init(CudaTensorIter *it, const Tensor *t);

// Device-side read/write with contiguity fast path
__device__ __forceinline__ float cuda_iter_read(const CudaTensorIter *it, u64 idx) {
  if (it->contiguous)
    return it->data[idx];
  u64 coords[MAX_NDIM];
  cuda_iter_linear_to_coords(idx, it->shape, it->ndim, coords);
  return it->data[cuda_iter_get_offset(coords, it->strides, it->ndim)];
}

__device__ __forceinline__ void cuda_iter_write(CudaTensorIter *it, u64 idx, float val) {
  if (it->contiguous) {
    it->data[idx] = val;
    return;
  }
  u64 coords[MAX_NDIM];
  cuda_iter_linear_to_coords(idx, it->shape, it->ndim, coords);
  it->data[cuda_iter_get_offset(coords, it->strides, it->ndim)] = val;
}

__device__ __forceinline__ void cuda_iter_add(CudaTensorIter *it, u64 idx, float val) {
  if (it->contiguous) {
    atomicAdd(&it->data[idx], val);
    return;
  }
  u64 coords[MAX_NDIM];
  cuda_iter_linear_to_coords(idx, it->shape, it->ndim, coords);
  atomicAdd(&it->data[cuda_iter_get_offset(coords, it->strides, it->ndim)], val);
}
