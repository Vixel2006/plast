#include "core/tensor.h"
#include "kernels/cpu/cpu_tensor_init.h"
#ifdef CUDA_AVAILABLE
#include "kernels/cuda/cuda_tensor_init.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int dtype_size(DTYPE dtype) {
  switch (dtype) {
  case INT32:
    return 4;
  case FLOAT32:
    return 4;
  default:
    return 0;
  }
}

u64 *compute_strides(const u64 shape[], u64 ndim) {
  u64 *strides = malloc(sizeof(u64) * ndim);
  strides[ndim - 1] = 1;
  for (u64 i = ndim - 1; i > 0; --i) {
    strides[i - 1] = strides[i] * shape[i];
  }
  return strides;
}

bool is_contiguous(const Tensor *t) {
  u64 expected_stride = 1;
  for (u64 i = t->ndim - 1; i > 0; --i) {
    if (t->strides[i] != expected_stride)
      return false;
    expected_stride *= t->shape[i];
  }
  return true;
}

u64 numel(const Tensor *t) {
  u64 num_elements = 1;
  for (u64 i = 0; i < t->ndim; ++i) {
    num_elements *= t->shape[i];
  }
  return num_elements;
}

Tensor *arena_tensor_alloc(Arena *meta_arena, Arena *data_arena, u64 shape[], u64 ndim,
                           u64 *strides, DTYPE dtype, bool requires_grad, struct Node *creator,
                           DEVICE device) {
  Tensor *t = (Tensor *)arena_alloc(meta_arena, sizeof(Tensor), 8);

  t->ndim = ndim;
  t->device = device;

  memcpy(t->shape, shape, ndim * sizeof(u64));
  memcpy(t->strides, strides, ndim * sizeof(u64));

  t->requires_grad = requires_grad;
  t->dtype = dtype;

  u64 num_elements = numel(t);
  t->data = (void *)arena_alloc(data_arena, num_elements * dtype_size(t->dtype), 8);

  if (t->requires_grad) {
    Tensor *grad = (Tensor *)arena_alloc(meta_arena, sizeof(Tensor), 8);
    grad->ndim = t->ndim;
    grad->device = device;
    memcpy(grad->shape, t->shape, ndim * sizeof(u64));
    memcpy(grad->strides, t->strides, ndim * sizeof(u64));
    grad->requires_grad = false;
    grad->dtype = dtype;

    grad->data = (void *)arena_alloc(data_arena, num_elements * dtype_size(grad->dtype), 8);
    if (device == CPU) {
      zeros_cpu(grad, num_elements);
    }
#ifdef CUDA_AVAILABLE
    else {
      zeros_cuda(grad, num_elements);
    }
#endif
    t->grad = grad;
  } else {
    t->grad = NULL;
  }

  t->creator = creator;

  return t;
}

void zeros(Tensor *t, u64 num_elements) {
  if (t->device == CPU) {
    zeros_cpu(t, num_elements);
  }
#ifdef CUDA_AVAILABLE
  else {
    zeros_cuda(t, num_elements);
  }
#endif
}

void ones(Tensor *t, u64 num_elements) {
  if (t->device == CPU) {
    ones_cpu(t, num_elements);
  }
#ifdef CUDA_AVAILABLE
  else {
    ones_cuda(t, num_elements);
  }
#endif
}

Tensor *init(Arena *meta_arena, Arena *data_arena, DEVICE device, DTYPE dtype, u64 shape[],
             u64 ndim, bool requires_grad, void (*init_fn)(Tensor *t, u64 num_elements)) {
  u64 *strides = compute_strides(shape, ndim);
  Tensor *t = arena_tensor_alloc(meta_arena, data_arena, shape, ndim, strides, dtype, requires_grad,
                                 NULL, device);
  if (!t)
    return NULL;

  t->device = device;
  u64 num_elements = numel(t);

  if (init_fn) {
    init_fn(t, num_elements);
  }

  free(strides);

  return t;
}

void set_ones_grad(Tensor *t) {
  if (t->device == CPU) {
    set_ones_grad_cpu(t);
  }
#ifdef CUDA_AVAILABLE
  else {
    set_ones_grad_cuda(t);
  }
#endif
}

// Generic tensor creation and freeing functions
Tensor *tensor_create(u64 *shape, u64 ndim, DTYPE dtype, DEVICE device) {
  Tensor *t = (Tensor *)malloc(sizeof(Tensor));
  if (t == NULL) {
    fprintf(stderr, "Failed to allocate memory for Tensor struct\n");
    return NULL;
  }

  t->ndim = ndim;
  t->dtype = dtype;
  t->device = device;
  t->requires_grad = false;
  t->creator = NULL;
  t->grad = NULL;

  memcpy(t->shape, shape, ndim * sizeof(u64));
  u64 *strides = compute_strides(shape, ndim);

  for (u64 i = 0; i < ndim; ++i)
    t->strides[i] = strides[i];

  free(strides);

  u64 num_elements = numel(t);
  t->data = malloc(num_elements * dtype_size(t->dtype));
  if (t->data == NULL) {
    fprintf(stderr, "Failed to allocate memory for Tensor data\n");
    free(t);
    return NULL;
  }
  // Initialize data to zeros
  memset(t->data, 0, num_elements * dtype_size(t->dtype));

  return t;
}

void tensor_free(Tensor *t) {
  if (t == NULL) {
    return;
  }
  free(t->data);
  if (t->grad != NULL) {
    tensor_free(t->grad); // Recursively free gradient tensor
  }
  free(t);
}
