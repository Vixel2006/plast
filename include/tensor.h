#pragma once

#include "arena.h"
#include <stdbool.h>

#define MAX_NDIM 8

typedef enum DTYPE { INT32, FLOAT32 } DTYPE;

#ifdef __cplusplus
extern "C" {
#endif

int dtype_size(DTYPE dtype);

#ifdef __cplusplus
}
#endif

typedef struct Tensor {
  struct Tensor *grad;
  struct Node *creator;
  void *data;
  u64 shape[MAX_NDIM];
  u64 strides[MAX_NDIM];
  u64 ndim;
  DTYPE dtype;
  DEVICE device;
  bool requires_grad;
} Tensor;

#include "kernels/cpu/cpu_tensor_init.h"
#include "kernels/cuda/cuda_tensor_init.h"

#ifdef __cplusplus
extern "C" {
#endif

u64 *compute_strides(const u64 shape[], u64 ndim);
bool is_contiguous(const Tensor *t);

u64 numel(const Tensor *t);

Tensor *arena_tensor_alloc(Arena *meta_arena, Arena *data_arena, u64 shape[], u64 ndim,
                           u64 *strides, DTYPE dtype, bool requires_grad, struct Node *creator,
                           DEVICE device);

void zeros(Tensor *t, u64 num_elements);
void ones(Tensor *t, u64 num_elements);
void set_ones_grad(Tensor *t);

Tensor *init(Arena *meta_arena, Arena *data_arena, DEVICE device, DTYPE dtype, u64 shape[],
             u64 ndim, bool requires_grad, void (*init_fn)(Tensor *t, u64 num_elements));

// Generic tensor creation and freeing functions
Tensor *tensor_create(u64 *shape, u64 ndim, DTYPE dtype, DEVICE device);
void tensor_free(Tensor *t);

#ifdef __cplusplus
}
#endif
