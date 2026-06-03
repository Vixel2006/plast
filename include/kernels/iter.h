#pragma once

#include "tensor.h"
#include <stdbool.h>

typedef struct {
  float *data;
  u64 numel;
  bool contiguous;
  u64 shape[MAX_NDIM];
  u64 strides[MAX_NDIM];
  u64 ndim;
} TensorIter;

void iter_init(TensorIter *it, const Tensor *t);
float iter_read(const TensorIter *it, u64 idx);
void iter_write(TensorIter *it, u64 idx, float val);
void iter_add(TensorIter *it, u64 idx, float val);

// Coordinate-based access (always works, never uses contiguity shortcut)
float iter_read_coords(const TensorIter *it, const u64 *coords);
void iter_write_coords(TensorIter *it, const u64 *coords, float val);
void iter_add_coords(TensorIter *it, const u64 *coords, float val);

// Linear-to-coords conversion
void iter_linear_to_coords(u64 idx, const u64 *shape, u64 ndim, u64 *coords);
u64 iter_get_offset(const u64 *coords, const u64 *strides, u64 ndim);
