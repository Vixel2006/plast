#include "cuda_runtime.h"
#include "definitions.h"
#include "kernels/cpu_utils.h"
#include "kernels/cuda/cuda_iter.cuh"
#include <cstring>

void cuda_iter_init(CudaTensorIter *it, const Tensor *t) {
  it->data = (float *)t->data;
  it->numel = 1;
  for (u64 i = 0; i < t->ndim; ++i)
    it->numel *= t->shape[i];
  it->contiguous = is_contiguous(t);
  memcpy(it->shape, t->shape, t->ndim * sizeof(u64));
  memcpy(it->strides, t->strides, t->ndim * sizeof(u64));
  it->ndim = t->ndim;
  it->dim_offset = 0;
}
