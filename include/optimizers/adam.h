#pragma once

#include "core/tensor.h"

typedef struct {
  float lr;
  float beta1;
  float beta2;
  float epsilon;
  int t;
  Tensor **m;
  Tensor **v;
  Arena *data_arena;
  Arena *optimizer_arena;
} Adam;

Adam alloc_adam(Arena *optimizer_arena, Arena *data_arena, float lr, float beta1, float beta2,
                float epsilon);

void adam_step_cpu(Adam *optimizer, Tensor **parameters, int num_parameters);

#ifdef __cplusplus
extern "C" {
#endif
void adam_step_cuda(Adam *optimizer, Tensor **parameters, int num_parameters);
#ifdef __cplusplus
}
#endif
