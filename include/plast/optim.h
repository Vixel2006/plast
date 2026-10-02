#pragma once

#include "core/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*plast_step_fn)(void *state, Tensor **params, int n);

typedef struct PlastOptimizer {
  void *state;
  plast_step_fn step_fn;
  void (*free_fn)(void *state);
} PlastOptimizer;

PlastOptimizer *plast_optim_sgd_create(float lr);
PlastOptimizer *plast_optim_adam_create(float lr, float beta1, float beta2, float epsilon);
PlastOptimizer *plast_optim_adamw_create(float lr, float beta1, float beta2, float epsilon,
                                         float weight_decay);
void plast_optim_free(PlastOptimizer *opt);

#ifdef __cplusplus
}
#endif
