#pragma once

#include "core/tensor.h"

typedef struct TensorDataset {
  Tensor **tensors;
  u64 num_tensors;
  u64 size;
} TensorDataset;

#ifdef __cplusplus
extern "C" {
#endif

TensorDataset *create_tensor_dataset(Tensor **tensors, u64 num_tensors);
void free_tensor_dataset(TensorDataset *dataset);

#ifdef __cplusplus
}
#endif
