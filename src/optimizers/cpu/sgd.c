#include "sgd.h"
#include "core/arena.h"
#include <stdio.h>
#include <stdlib.h>

SGD arena_alloc_sgd(float lr) {
  SGD optim = {.lr = lr};
  return optim;
}

void sgd_step_cpu(SGD *optimizer, Tensor **parameters, int num_parameters) {
  if (optimizer == NULL) {
    fprintf(stderr, "SGD is NULL\n");
    return;
  }

  for (int i = 0; i < num_parameters; ++i) {
    Tensor *param = parameters[i];
    if (param == NULL || param->grad == NULL) {
      continue;
    }

    // Assuming data and grad are flat float arrays
    float *data = (float *)param->data;
    float *grad = (float *)param->grad->data;

    for (size_t j = 0; j < numel(param); ++j) {
      data[j] -= optimizer->lr * grad[j];
    }
  }
}
