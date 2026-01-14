#pragma once

#include "definitions.h"
#include "tensor.h"

__device__ inline u64 cuda_get_offset(const u64 *coords, const u64 *strides,
                                      u64 ndim) {
  u64 offset = 0;
  for (u64 i = 0; i < ndim; ++i) {
    offset += coords[i] * strides[i];
  }
  return offset;
}

__device__ inline u64 cuda_get_offset_broadcast(const u64 *coords, u64 ndim,
                                                const u64 *t_strides,
                                                const u64 *t_shape, u64 t_ndim) {
  u64 offset = 0;
  int dim_offset = (int)ndim - (int)t_ndim;
  for (u64 i = 0; i < t_ndim; ++i) {
    u64 coord = coords[i + dim_offset];
    if (t_shape[i] == 1) {
      coord = 0;
    }
    offset += coord * t_strides[i];
  }
  return offset;
}

__device__ __forceinline__ u64 cuda_numel_from_shape(const u64 *shape, u64 ndim) {
  u64 numel = 1;
  for (u64 i = 0; i < ndim; ++i) {
    numel *= shape[i];
  }
  return numel;
}

__device__ inline void cuda_linear_to_coords(u64 linear_idx, const u64 *shape,
                                             u64 ndim, u64 *coords) {
  for (u64 i = ndim; i-- > 0;) {
    coords[i] = linear_idx % shape[i];
    linear_idx /= shape[i];
  }
}

