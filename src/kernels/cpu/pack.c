#include "kernels/pack.h"
#include "kernels/cpu_utils.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void pack_to_buf(const Tensor *src, void *dst) {
  u64 num_elements = numel(src);
  u64 element_size = dtype_size(src->dtype);

  if (is_contiguous(src)) {
    memcpy(dst, src->data, num_elements * element_size);
    return;
  }

  u64 *coords = (u64 *)malloc(src->ndim * sizeof(u64));
  if (!coords)
    return;

  for (u64 i = 0; i < num_elements; ++i) {
    linear_to_coords(i, src->shape, src->ndim, coords);
    u64 src_offset = get_offset(coords, src->strides, src->ndim);
    memcpy((u8 *)dst + i * element_size, (u8 *)src->data + src_offset * element_size, element_size);
  }

  free(coords);
}

void tensor_pack_init(TensorPack *p, const Tensor *t) {
  p->data = NULL;
  p->_buf = NULL;

  if (is_contiguous(t)) {
    p->data = t->data;
    return;
  }

  u64 num_bytes = numel(t) * dtype_size(t->dtype);
  p->_buf = malloc(num_bytes);
  if (!p->_buf) {
    fprintf(stderr, "tensor_pack_init: failed to allocate %llu bytes\n",
            (unsigned long long)num_bytes);
    return;
  }
  pack_to_buf(t, p->_buf);
  p->data = p->_buf;
}

void tensor_pack_release(TensorPack *p) {
  free(p->_buf);
  p->data = NULL;
  p->_buf = NULL;
}
