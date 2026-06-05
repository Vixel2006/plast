#pragma once

#include "core/tensor.h"

typedef struct {
  float lr;
} SGD;

SGD arena_alloc_sgd(float lr);
void sgd_step_cpu(SGD *optimizer, Tensor **parameters, int num_parameters);

#ifdef __cplusplus
extern "C" {
#endif
void sgd_step_cuda(SGD *optimizer, Tensor **parameters, int num_parameters);
#ifdef __cplusplus
}
#endif
