#pragma once

#include "core/tensor.h"
#include "core/graph.h"
#include "core/arena.h"
#include "core/op.h"
#include "core/node.h"
#include "scheduler/scheduler.h"
#include "optimizers/sgd.h"
#include "optimizers/adam.h"
#include "optimizers/adamw.h"
#include "optimizers/zero_grad.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PLAST_MAX_LAYERS 64
#define PLAST_MAX_PARAMS 256
#define PLAST_MAX_NAME 128

typedef enum {
  PLAST_ACT_NONE,
  PLAST_ACT_RELU,
  PLAST_ACT_LEAKY_RELU,
  PLAST_ACT_SIGMOID,
  PLAST_ACT_TANH,
  PLAST_ACT_SOFTMAX,
} PlastActivation;

typedef enum {
  PLAST_LAYER_DENSE,
  PLAST_LAYER_CONV2D,
  PLAST_LAYER_ACTIVATION,
  PLAST_LAYER_FLATTEN,
  PLAST_LAYER_DROPOUT,
  PLAST_LAYER_BATCHNORM1D,
  PLAST_LAYER_BATCHNORM2D,
  PLAST_LAYER_LAYERNORM,
} PlastLayerType;

typedef struct {
  PlastLayerType type;
  union {
    struct {
      u64 out_features;
      bool bias;
    } dense;
    struct {
      u64 out_channels;
      u64 kernel_size;
      u64 stride;
      bool bias;
    } conv2d;
    struct {
      PlastActivation act;
      float leaky_slope;
    } activation;
    struct {
      float p;
    } dropout;
    struct {
      u64 num_features;
      float eps;
      float momentum;
    } batchnorm;
    struct {
      u64 normalized_shape;
      float eps;
    } layernorm;
  };
} PlastLayerDesc;

typedef struct {
  PlastLayerDesc desc;
  u64 output_ndim;
  u64 output_shape[MAX_NDIM];
  Tensor *output;
  Tensor *weight;
  Tensor *bias;
  int param_start;
  int num_params;
} PlastLayer;

typedef struct PlastOptimizer PlastOptimizer;

typedef struct PlastModel {
  DEVICE device;
  bool compiled;

  PlastLayer layers[PLAST_MAX_LAYERS];
  int num_layers;

  u64 input_ndim;
  u64 input_shape[MAX_NDIM];
  Tensor *input_tensor;

  Arena meta;
  Arena data;

  Tensor *params[PLAST_MAX_PARAMS];
  char param_names[PLAST_MAX_PARAMS][PLAST_MAX_NAME];
  int num_params;

  Node *last_node;
  Tensor *output_tensor;

  Scheduler *scheduler;
  PlastOptimizer *optimizer;
} PlastModel;

typedef void (*plast_init_fn)(Tensor *t, u64 n);

PlastModel *plast_model_create(DEVICE device);
void plast_model_free(PlastModel *m);

void plast_model_add_dense(PlastModel *m, u64 out_features, bool bias);
void plast_model_add_conv2d(PlastModel *m, u64 out_channels, u64 kernel_size, u64 stride,
                            bool bias);
void plast_model_add_activation(PlastModel *m, PlastActivation act);
void plast_model_add_leaky_relu(PlastModel *m, float negative_slope);
void plast_model_add_flatten(PlastModel *m);
void plast_model_add_dropout(PlastModel *m, float p);
void plast_model_add_batchnorm1d(PlastModel *m, u64 num_features, float eps, float momentum);
void plast_model_add_batchnorm2d(PlastModel *m, u64 num_features, float eps, float momentum);
void plast_model_add_layernorm(PlastModel *m, u64 normalized_shape, float eps);

void plast_model_compile(PlastModel *m, const u64 *input_shape, u64 input_ndim);

Tensor *plast_model_forward(PlastModel *m);
void plast_model_zero_grad(PlastModel *m);
void plast_model_backward(PlastModel *m, Tensor *loss);
void plast_optimizer_step(PlastModel *m);

Tensor *plast_model_input(PlastModel *m);
Tensor *plast_model_output(PlastModel *m);
Arena *plast_model_meta_arena(PlastModel *m);
Arena *plast_model_data_arena(PlastModel *m);

Tensor *plast_tensor_from_array(const float *data, const u64 *shape, u64 ndim);
Tensor *plast_scalar(float value, DEVICE device);
void plast_tensor_print(const Tensor *t, const char *name);

void plast_model_save(const PlastModel *m, const char *path);
PlastModel *plast_model_load(const char *path, DEVICE device);

void plast_model_summary(const PlastModel *m);
void plast_model_set_weight_init(PlastModel *m, int layer_idx, plast_init_fn fn);
void plast_model_print_weights(const PlastModel *m);
void plast_model_set_optimizer(PlastModel *m, PlastOptimizer *opt);

#ifdef __cplusplus
}
#endif
