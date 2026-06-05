#pragma once

#include "core/tensor.h"

typedef struct {
  float lr;
  float beta1;
  float beta2;
  float epsilon;
  float weight_decay;
  int t;
  Tensor **m;
  Tensor **v;
  Arena *data_arena;
  Arena *optimizer_arena;
} AdamW;

AdamW adamw_alloc(Arena *optimizer_arena, Arena *data_arena, float lr, float beta1, float beta2,
                  float epsilon, float weight_decay);

void adamw_step_cpu(AdamW *optimizer, Tensor **parameters, int num_parameters);

#ifdef __cplusplus
extern "C" {
#endif
void adamw_step_cuda(AdamW *optimizer, Tensor **parameters, int num_parameters);
#ifdef __cplusplus
}
#endif
