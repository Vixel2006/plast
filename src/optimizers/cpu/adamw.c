#include "adamw.h"
#include "arena.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

AdamW adamw_alloc(Arena *optimizer_arena, Arena *data_arena, float lr, float beta1, float beta2,
                  float epsilon, float weight_decay) {
  AdamW adamw = {.lr = lr,
                 .beta1 = beta1,
                 .beta2 = beta2,
                 .epsilon = epsilon,
                 .weight_decay = weight_decay,
                 .data_arena = data_arena,
                 .optimizer_arena = optimizer_arena};
  return adamw;
}

void adamw_step_cpu(AdamW *optimizer, Tensor **parameters, int num_parameters) {
  if (optimizer == NULL) {
    fprintf(stderr, "AdamW is NULL\n");
    return;
  }

  optimizer->t++;

  if (optimizer->m == NULL) {
    optimizer->m = (Tensor **)arena_alloc(optimizer->optimizer_arena,
                                          sizeof(Tensor *) * num_parameters, _Alignof(Tensor *));
    optimizer->v = (Tensor **)arena_alloc(optimizer->optimizer_arena,
                                          sizeof(Tensor *) * num_parameters, _Alignof(Tensor *));
    if (optimizer->m == NULL || optimizer->v == NULL) {
      fprintf(stderr, "Failed to allocate memory for AdamW moment vectors\n");
      return;
    }
    for (int i = 0; i < num_parameters; ++i) {
      optimizer->m[i] =
          init(optimizer->optimizer_arena, optimizer->data_arena, parameters[i]->device,
               parameters[i]->dtype, parameters[i]->shape, parameters[i]->ndim, false, zeros);
      optimizer->v[i] =
          init(optimizer->optimizer_arena, optimizer->data_arena, parameters[i]->device,
               parameters[i]->dtype, parameters[i]->shape, parameters[i]->ndim, false, zeros);
    }
  }

  float lr_t = optimizer->lr * sqrt(1.0f - powf(optimizer->beta2, optimizer->t)) /
               (1.0f - powf(optimizer->beta1, optimizer->t));

  for (int i = 0; i < num_parameters; ++i) {
    Tensor *param = parameters[i];
    if (param == NULL || param->grad == NULL) {
      continue;
    }

    float *data = (float *)param->data;
    float *grad = (float *)param->grad->data;
    float *m_data = (float *)optimizer->m[i]->data;
    float *v_data = (float *)optimizer->v[i]->data;

    for (size_t j = 0; j < numel(param); ++j) {
      // Apply weight decay
      data[j] -= optimizer->lr * optimizer->weight_decay * data[j];

      m_data[j] = optimizer->beta1 * m_data[j] + (1.0f - optimizer->beta1) * grad[j];
      v_data[j] = optimizer->beta2 * v_data[j] + (1.0f - optimizer->beta2) * grad[j] * grad[j];
      data[j] -= lr_t * m_data[j] / (sqrtf(v_data[j]) + optimizer->epsilon);
    }
  }
}
