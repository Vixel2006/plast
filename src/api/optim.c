#include "plast/optim.h"
#include "optimizers/sgd.h"
#include "optimizers/adam.h"
#include "optimizers/adamw.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
  SGD sgd;
} SgdState;

static void sgd_step_fn(void *state, Tensor **params, int n) {
  SgdState *s = (SgdState *)state;
  sgd_step_cpu(&s->sgd, params, n);
}

static void sgd_free_fn(void *state) {
  free(state);
}

typedef struct {
  Adam adam;
} AdamState;

static void adam_step_fn(void *state, Tensor **params, int n) {
  AdamState *s = (AdamState *)state;
  adam_step_cpu(&s->adam, params, n);
}

static void adam_free_fn(void *state) {
  free(state);
}

typedef struct {
  AdamW adamw;
} AdamWState;

static void adamw_step_fn(void *state, Tensor **params, int n) {
  AdamWState *s = (AdamWState *)state;
  adamw_step_cpu(&s->adamw, params, n);
}

static void adamw_free_fn(void *state) {
  free(state);
}

PlastOptimizer *plast_optim_sgd_create(float lr) {
  PlastOptimizer *opt = (PlastOptimizer *)malloc(sizeof(PlastOptimizer));
  if (!opt)
    return NULL;
  SgdState *s = (SgdState *)malloc(sizeof(SgdState));
  if (!s) {
    free(opt);
    return NULL;
  }
  memset(s, 0, sizeof(SgdState));
  s->sgd.lr = lr;
  opt->state = s;
  opt->step_fn = sgd_step_fn;
  opt->free_fn = sgd_free_fn;
  return opt;
}

PlastOptimizer *plast_optim_adam_create(float lr, float beta1, float beta2, float epsilon) {
  PlastOptimizer *opt = (PlastOptimizer *)malloc(sizeof(PlastOptimizer));
  if (!opt)
    return NULL;
  AdamState *s = (AdamState *)calloc(1, sizeof(AdamState));
  if (!s) {
    free(opt);
    return NULL;
  }
  // Adam state needs arena pointers that will be set during compile
  // For now, initialize with dummy arenas — the user must call init
  s->adam.lr = lr;
  s->adam.beta1 = beta1;
  s->adam.beta2 = beta2;
  s->adam.epsilon = epsilon;
  s->adam.t = 0;
  s->adam.m = NULL;
  s->adam.v = NULL;
  s->adam.data_arena = NULL;
  s->adam.optimizer_arena = NULL;
  opt->state = s;
  opt->step_fn = adam_step_fn;
  opt->free_fn = adam_free_fn;
  return opt;
}

PlastOptimizer *plast_optim_adamw_create(float lr, float beta1, float beta2, float epsilon,
                                         float weight_decay) {
  PlastOptimizer *opt = (PlastOptimizer *)malloc(sizeof(PlastOptimizer));
  if (!opt)
    return NULL;
  AdamWState *s = (AdamWState *)calloc(1, sizeof(AdamWState));
  if (!s) {
    free(opt);
    return NULL;
  }
  s->adamw.lr = lr;
  s->adamw.beta1 = beta1;
  s->adamw.beta2 = beta2;
  s->adamw.epsilon = epsilon;
  s->adamw.weight_decay = weight_decay;
  s->adamw.t = 0;
  s->adamw.m = NULL;
  s->adamw.v = NULL;
  s->adamw.data_arena = NULL;
  s->adamw.optimizer_arena = NULL;
  opt->state = s;
  opt->step_fn = adamw_step_fn;
  opt->free_fn = adamw_free_fn;
  return opt;
}

void plast_optim_free(PlastOptimizer *opt) {
  if (!opt)
    return;
  if (opt->free_fn)
    opt->free_fn(opt->state);
  free(opt);
}
