#include "kernels/iter.h"
#include "kernels/cpu_utils.h"
#include <string.h>

void iter_init(TensorIter *it, const Tensor *t) {
  it->data = (float *)t->data;
  it->numel = 1;
  for (u64 i = 0; i < t->ndim; ++i)
    it->numel *= t->shape[i];
  it->contiguous = is_contiguous(t);
  memcpy(it->shape, t->shape, t->ndim * sizeof(u64));
  memcpy(it->strides, t->strides, t->ndim * sizeof(u64));
  it->ndim = t->ndim;
}

float iter_read(const TensorIter *it, u64 idx) {
  if (it->contiguous)
    return it->data[idx];
  u64 coords[MAX_NDIM];
  iter_linear_to_coords(idx, it->shape, it->ndim, coords);
  return it->data[iter_get_offset(coords, it->strides, it->ndim)];
}

void iter_write(TensorIter *it, u64 idx, float val) {
  if (it->contiguous) {
    it->data[idx] = val;
    return;
  }
  u64 coords[MAX_NDIM];
  iter_linear_to_coords(idx, it->shape, it->ndim, coords);
  it->data[iter_get_offset(coords, it->strides, it->ndim)] = val;
}

void iter_add(TensorIter *it, u64 idx, float val) {
  if (it->contiguous) {
    it->data[idx] += val;
    return;
  }
  u64 coords[MAX_NDIM];
  iter_linear_to_coords(idx, it->shape, it->ndim, coords);
  it->data[iter_get_offset(coords, it->strides, it->ndim)] += val;
}

float iter_read_coords(const TensorIter *it, const u64 *coords) {
  return it->data[iter_get_offset(coords, it->strides, it->ndim)];
}

void iter_write_coords(TensorIter *it, const u64 *coords, float val) {
  it->data[iter_get_offset(coords, it->strides, it->ndim)] = val;
}

void iter_add_coords(TensorIter *it, const u64 *coords, float val) {
  it->data[iter_get_offset(coords, it->strides, it->ndim)] += val;
}

void iter_linear_to_coords(u64 idx, const u64 *shape, u64 ndim, u64 *coords) {
  for (u64 i = ndim; i-- > 0;) {
    coords[i] = idx % shape[i];
    idx /= shape[i];
  }
}

u64 iter_get_offset(const u64 *coords, const u64 *strides, u64 ndim) {
  u64 offset = 0;
  for (u64 i = 0; i < ndim; ++i)
    offset += coords[i] * strides[i];
  return offset;
}
