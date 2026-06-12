#include "plast/model.h"
#include "plast/optim.h"
#include "kernels/ops/shape.h"
#include "kernels/cpu_utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define META_SIZE Mib(64)
#define DATA_SIZE Mib(512)

static void kaiming_init(Tensor *t, u64 n) {
  float scale = sqrtf(2.0f / (float)t->shape[0]);
  float *buf = (float *)malloc(n * sizeof(float));
  for (u64 i = 0; i < n; ++i)
    buf[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
  memcpy(t->data, buf, n * sizeof(float));
  free(buf);
}

static void zero_init(Tensor *t, u64 n) {
  memset(t->data, 0, n * sizeof(float));
}

static u64 product(const u64 *shape, u64 ndim) {
  u64 p = 1;
  for (u64 i = 0; i < ndim; ++i)
    p *= shape[i];
  return p;
}

static Tensor *alloc_tensor(PlastModel *m, const u64 *shape, u64 ndim, bool requires_grad,
                            plast_init_fn init_fn) {
  Tensor *t = init(&m->meta, &m->data, m->device, FLOAT32, (u64 *)shape, ndim, requires_grad, NULL);
  if (init_fn)
    init_fn(t, numel(t));
  return t;
}

static Tensor *add_node(PlastModel *m, Tensor **inputs, int num_inputs, const u64 *shape, u64 ndim,
                        OP_TYPE op_type, KernelParams params) {
  bool requires_grad = false;
  for (int i = 0; i < num_inputs; ++i) {
    if (inputs[i]->requires_grad) {
      requires_grad = true;
      break;
    }
  }
  // Copy input pointers into arena (node->inputs must point to arena memory)
  Tensor **inputs_copy = (Tensor **)arena_alloc(&m->meta, num_inputs * sizeof(Tensor *), 8);
  for (int i = 0; i < num_inputs; ++i)
    inputs_copy[i] = inputs[i];

  Tensor *output = alloc_tensor(m, shape, ndim, requires_grad, NULL);
  Node *n = arena_node_alloc(&m->meta, inputs_copy, num_inputs, output, get_op_impl(op_type),
                             op_type, params);
  (void)n;
  return output;
}

static Tensor *add_constant(PlastModel *m, float value) {
  Tensor *t = alloc_tensor(m, (u64[]){1}, 1, false, NULL);
  float *ptr = (float *)t->data;
  ptr[0] = value;
  return t;
}

PlastModel *plast_model_create(DEVICE device) {
  PlastModel *m = (PlastModel *)calloc(1, sizeof(PlastModel));
  if (!m)
    return NULL;
  m->device = device;
  m->compiled = false;
  m->num_layers = 0;
  m->num_params = 0;
  m->last_node = NULL;
  m->output_tensor = NULL;
  m->scheduler = NULL;
  m->optimizer = NULL;
  return m;
}

void plast_model_free(PlastModel *m) {
  if (!m)
    return;
  if (m->scheduler)
    scheduler_release(m->scheduler);
  if (m->optimizer)
    plast_optim_free(m->optimizer);
  arena_release(&m->meta);
  arena_release(&m->data);
  free(m);
}

// --- Layer registration ---

void plast_model_add_dense(PlastModel *m, u64 out_features, bool bias) {
  if (m->compiled) {
    fprintf(stderr, "plast: cannot add layers after compilation\n");
    return;
  }
  if (m->num_layers >= PLAST_MAX_LAYERS) {
    fprintf(stderr, "plast: max layers (%d) reached\n", PLAST_MAX_LAYERS);
    return;
  }
  PlastLayer *l = &m->layers[m->num_layers++];
  memset(l, 0, sizeof(PlastLayer));
  l->desc.type = PLAST_LAYER_DENSE;
  l->desc.dense.out_features = out_features;
  l->desc.dense.bias = bias;
}

void plast_model_add_conv2d(PlastModel *m, u64 out_channels, u64 kernel_size, u64 stride,
                            bool bias) {
  if (m->compiled) {
    fprintf(stderr, "plast: cannot add layers after compilation\n");
    return;
  }
  if (m->num_layers >= PLAST_MAX_LAYERS) {
    fprintf(stderr, "plast: max layers (%d) reached\n", PLAST_MAX_LAYERS);
    return;
  }
  PlastLayer *l = &m->layers[m->num_layers++];
  memset(l, 0, sizeof(PlastLayer));
  l->desc.type = PLAST_LAYER_CONV2D;
  l->desc.conv2d.out_channels = out_channels;
  l->desc.conv2d.kernel_size = kernel_size;
  l->desc.conv2d.stride = stride;
  l->desc.conv2d.bias = bias;
}

void plast_model_add_activation(PlastModel *m, PlastActivation act) {
  if (m->compiled)
    return;
  if (m->num_layers >= PLAST_MAX_LAYERS)
    return;
  PlastLayer *l = &m->layers[m->num_layers++];
  memset(l, 0, sizeof(PlastLayer));
  l->desc.type = PLAST_LAYER_ACTIVATION;
  l->desc.activation.act = act;
  l->desc.activation.leaky_slope = 0.01f;
}

void plast_model_add_leaky_relu(PlastModel *m, float negative_slope) {
  if (m->compiled)
    return;
  if (m->num_layers >= PLAST_MAX_LAYERS)
    return;
  PlastLayer *l = &m->layers[m->num_layers++];
  memset(l, 0, sizeof(PlastLayer));
  l->desc.type = PLAST_LAYER_ACTIVATION;
  l->desc.activation.act = PLAST_ACT_LEAKY_RELU;
  l->desc.activation.leaky_slope = negative_slope;
}

void plast_model_add_flatten(PlastModel *m) {
  if (m->compiled)
    return;
  if (m->num_layers >= PLAST_MAX_LAYERS)
    return;
  PlastLayer *l = &m->layers[m->num_layers++];
  memset(l, 0, sizeof(PlastLayer));
  l->desc.type = PLAST_LAYER_FLATTEN;
}

void plast_model_add_dropout(PlastModel *m, float p) {
  if (m->compiled)
    return;
  if (m->num_layers >= PLAST_MAX_LAYERS)
    return;
  PlastLayer *l = &m->layers[m->num_layers++];
  memset(l, 0, sizeof(PlastLayer));
  l->desc.type = PLAST_LAYER_DROPOUT;
  l->desc.dropout.p = p;
}

void plast_model_add_batchnorm1d(PlastModel *m, u64 num_features, float eps, float momentum) {
  if (m->compiled)
    return;
  if (m->num_layers >= PLAST_MAX_LAYERS)
    return;
  PlastLayer *l = &m->layers[m->num_layers++];
  memset(l, 0, sizeof(PlastLayer));
  l->desc.type = PLAST_LAYER_BATCHNORM1D;
  l->desc.batchnorm.num_features = num_features;
  l->desc.batchnorm.eps = eps;
  l->desc.batchnorm.momentum = momentum;
}

void plast_model_add_batchnorm2d(PlastModel *m, u64 num_features, float eps, float momentum) {
  if (m->compiled)
    return;
  if (m->num_layers >= PLAST_MAX_LAYERS)
    return;
  PlastLayer *l = &m->layers[m->num_layers++];
  memset(l, 0, sizeof(PlastLayer));
  l->desc.type = PLAST_LAYER_BATCHNORM2D;
  l->desc.batchnorm.num_features = num_features;
  l->desc.batchnorm.eps = eps;
  l->desc.batchnorm.momentum = momentum;
}

void plast_model_add_layernorm(PlastModel *m, u64 normalized_shape, float eps) {
  if (m->compiled)
    return;
  if (m->num_layers >= PLAST_MAX_LAYERS)
    return;
  PlastLayer *l = &m->layers[m->num_layers++];
  memset(l, 0, sizeof(PlastLayer));
  l->desc.type = PLAST_LAYER_LAYERNORM;
  l->desc.layernorm.normalized_shape = normalized_shape;
  l->desc.layernorm.eps = eps;
}

// --- Compilation ---

static void register_param(PlastModel *m, Tensor *t, const char *name) {
  if (m->num_params >= PLAST_MAX_PARAMS)
    return;
  m->params[m->num_params] = t;
  snprintf(m->param_names[m->num_params], PLAST_MAX_NAME, "%s", name);
  m->num_params++;
}

static void compile_layer(PlastModel *m, int idx, const u64 *in_shape, u64 in_ndim,
                          Tensor *input_tensor) {
  PlastLayer *l = &m->layers[idx];
  u64 out_shape[MAX_NDIM] = {0};
  u64 out_ndim = 0;

  switch (l->desc.type) {
  case PLAST_LAYER_DENSE: {
    u64 in_features = in_shape[in_ndim - 1];
    u64 out_features = l->desc.dense.out_features;
    u64 batch = product(in_shape, in_ndim - 1);

    // weight: [in_features, out_features]
    Tensor *w = alloc_tensor(m, (u64[]){in_features, out_features}, 2, true, kaiming_init);
    l->weight = w;
    l->param_start = m->num_params;
    l->num_params = 1;

    char wname[PLAST_MAX_NAME];
    snprintf(wname, sizeof(wname), "dense_%d.weight", idx);
    register_param(m, w, wname);

    // matmul: input @ weight
    u64 mm_shape[] = {batch, out_features};
    Tensor *mm_out = add_node(m, (Tensor *[]){input_tensor, w}, 2, mm_shape, 2, MATMUL,
                              (KernelParams){0, 0, 0.0f});

    if (l->desc.dense.bias) {
      // bias: [1, out_features]
      Tensor *b = alloc_tensor(m, (u64[]){1, out_features}, 2, true, zero_init);
      l->bias = b;
      l->num_params = 2;

      char bname[PLAST_MAX_NAME];
      snprintf(bname, sizeof(bname), "dense_%d.bias", idx);
      register_param(m, b, bname);

      Tensor *add_out = add_node(m, (Tensor *[]){mm_out, b}, 2, mm_shape, 2, ADD,
                                 (KernelParams){0, 0, 0.0f});
      l->output = add_out;
    } else {
      l->output = mm_out;
    }

    out_ndim = 2;
    out_shape[0] = batch;
    out_shape[1] = out_features;
    break;
  }

  case PLAST_LAYER_CONV2D: {
    // input: [N, C_in, H, W]
    u64 N = in_shape[0];
    u64 C_in = in_shape[1];
    u64 H = in_shape[2];
    u64 W = in_shape[3];
    u64 K = l->desc.conv2d.kernel_size;
    u64 S = l->desc.conv2d.stride;
    u64 C_out = l->desc.conv2d.out_channels;

    u64 H_out = (H - K) / S + 1;
    u64 W_out = (W - K) / S + 1;

    // weight: [C_out, C_in, K, K]
    u64 fan_in = C_in * K * K;
    Tensor *w = alloc_tensor(m, (u64[]){C_out, C_in, K, K}, 4, true, NULL);
    {
      float scale = sqrtf(2.0f / (float)fan_in);
      u64 n = numel(w);
      float *buf = (float *)malloc(n * sizeof(float));
      for (u64 i = 0; i < n; ++i)
        buf[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
      memcpy(w->data, buf, n * sizeof(float));
      free(buf);
    }
    l->weight = w;
    l->param_start = m->num_params;
    l->num_params = 1;

    char wname[PLAST_MAX_NAME];
    snprintf(wname, sizeof(wname), "conv_%d.weight", idx);
    register_param(m, w, wname);

    // conv2d output shape: [N*H_out*W_out, C_out] (raw op output)
    u64 conv_shape[] = {N * H_out * W_out, C_out};
    Tensor *conv_out =
        add_node(m, (Tensor *[]){input_tensor, w}, 2, conv_shape, 2, CONV2D,
                 (KernelParams){(u64)&m->meta, l->desc.conv2d.stride, 0.0f});

    // view: [N*H_out*W_out, C_out] → [N, H_out, W_out, C_out]
    u64 view_shape[] = {N, H_out, W_out, C_out};
    Tensor *view_out = add_node(m, (Tensor *[]){conv_out}, 1, view_shape, 4, VIEW,
                                (KernelParams){0, 0, 0.0f});

    // transpose(1, 3): [N, H_out, W_out, C_out] → [N, C_out, W_out, H_out]
    u64 t1_shape[] = {N, C_out, W_out, H_out};
    Tensor *t1_out = add_node(m, (Tensor *[]){view_out}, 1, t1_shape, 4, TRANSPOSE,
                              (KernelParams){1, 3, 0.0f});

    // transpose(2, 3): [N, C_out, W_out, H_out] → [N, C_out, H_out, W_out]
    u64 final_shape[] = {N, C_out, H_out, W_out};
    Tensor *final = add_node(m, (Tensor *[]){t1_out}, 1, final_shape, 4, TRANSPOSE,
                             (KernelParams){2, 3, 0.0f});

    if (l->desc.conv2d.bias) {
      Tensor *b = alloc_tensor(m, (u64[]){C_out}, 1, true, zero_init);
      l->bias = b;
      l->num_params = 2;

      char bname[PLAST_MAX_NAME];
      snprintf(bname, sizeof(bname), "conv_%d.bias", idx);
      register_param(m, b, bname);

      Tensor *b_view = add_node(m, (Tensor *[]){b}, 1, (u64[]){1, C_out, 1, 1}, 4, VIEW,
                                (KernelParams){0, 0, 0.0f});
      Tensor *add_out = add_node(m, (Tensor *[]){final, b_view}, 2, final_shape, 4, ADD,
                                 (KernelParams){0, 0, 0.0f});
      l->output = add_out;
    } else {
      l->output = final;
    }

    out_ndim = 4;
    out_shape[0] = N;
    out_shape[1] = C_out;
    out_shape[2] = H_out;
    out_shape[3] = W_out;
    break;
  }

  case PLAST_LAYER_ACTIVATION: {
    out_ndim = in_ndim;
    memcpy(out_shape, in_shape, in_ndim * sizeof(u64));

    switch (l->desc.activation.act) {
    case PLAST_ACT_RELU:
      l->output = add_node(m, (Tensor *[]){input_tensor}, 1, in_shape, in_ndim, LEAKY_RELU,
                           (KernelParams){0, 0, 0.0f});
      break;
    case PLAST_ACT_LEAKY_RELU:
      l->output = add_node(m, (Tensor *[]){input_tensor}, 1, in_shape, in_ndim, LEAKY_RELU,
                           (KernelParams){0, 0, l->desc.activation.leaky_slope});
      break;
    case PLAST_ACT_SIGMOID: {
      // sigmoid = 1 / (1 + exp(-x))
      Tensor *neg = add_node(m, (Tensor *[]){input_tensor}, 1, in_shape, in_ndim, NEG,
                             (KernelParams){0, 0, 0.0f});
      Tensor *exp = add_node(m, (Tensor *[]){neg}, 1, in_shape, in_ndim, EXP,
                             (KernelParams){0, 0, 0.0f});
      Tensor *one = add_constant(m, 1.0f);
      Tensor *add = add_node(m, (Tensor *[]){exp, one}, 2, in_shape, in_ndim, ADD,
                             (KernelParams){0, 0, 0.0f});
      l->output = add_node(m, (Tensor *[]){one, add}, 2, in_shape, in_ndim, DIV,
                           (KernelParams){0, 0, 0.0f});
      break;
    }
    case PLAST_ACT_TANH: {
      // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
      Tensor *two = add_constant(m, 2.0f);
      Tensor *two_x = add_node(m, (Tensor *[]){input_tensor, two}, 2, in_shape, in_ndim, MUL,
                               (KernelParams){0, 0, 0.0f});
      Tensor *exp_2x = add_node(m, (Tensor *[]){two_x}, 1, in_shape, in_ndim, EXP,
                                (KernelParams){0, 0, 0.0f});
      Tensor *one = add_constant(m, 1.0f);
      Tensor *add = add_node(m, (Tensor *[]){exp_2x, one}, 2, in_shape, in_ndim, ADD,
                             (KernelParams){0, 0, 0.0f});
      Tensor *sub = add_node(m, (Tensor *[]){exp_2x, one}, 2, in_shape, in_ndim, SUB,
                             (KernelParams){0, 0, 0.0f});
      l->output = add_node(m, (Tensor *[]){sub, add}, 2, in_shape, in_ndim, DIV,
                           (KernelParams){0, 0, 0.0f});
      break;
    }
    case PLAST_ACT_SOFTMAX: {
      // softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
      // reduce along last dim
      u64 last_dim = in_ndim - 1;
      u64 red_shape[MAX_NDIM];
      u64 red_ndim = in_ndim;
      memcpy(red_shape, in_shape, in_ndim * sizeof(u64));
      red_shape[last_dim] = 1;

      Tensor *max = add_node(m, (Tensor *[]){input_tensor}, 1, red_shape, red_ndim, MAX,
                             (KernelParams){last_dim, 1, 0.0f});
      Tensor *shifted = add_node(m, (Tensor *[]){input_tensor, max}, 2, in_shape, in_ndim, SUB,
                                 (KernelParams){0, 0, 0.0f});
      Tensor *exp = add_node(m, (Tensor *[]){shifted}, 1, in_shape, in_ndim, EXP,
                             (KernelParams){0, 0, 0.0f});
      Tensor *sum = add_node(m, (Tensor *[]){exp}, 1, red_shape, red_ndim, SUM,
                             (KernelParams){last_dim, 1, 0.0f});
      l->output = add_node(m, (Tensor *[]){exp, sum}, 2, in_shape, in_ndim, DIV,
                           (KernelParams){0, 0, 0.0f});
      break;
    }
    default:
      l->output = input_tensor;
      break;
    }
    break;
  }

  case PLAST_LAYER_FLATTEN: {
    u64 batch = in_shape[0];
    u64 flat = product(in_shape, in_ndim) / batch;
    out_ndim = 2;
    out_shape[0] = batch;
    out_shape[1] = flat;
    l->output = add_node(m, (Tensor *[]){input_tensor}, 1, out_shape, out_ndim, VIEW,
                         (KernelParams){0, 0, 0.0f});
    break;
  }

  case PLAST_LAYER_DROPOUT: {
    // dropout = x * mask (mask is random [0,1] > p, scaled by 1/(1-p))
    // For simplicity at compile time, just pass through during inference.
    // During training, the user manages the mask externally.

    // TODO: proper dropout with random mask generation
    // For now, identity during compile — handled at runtime
    l->output = input_tensor;
    out_ndim = in_ndim;
    memcpy(out_shape, in_shape, in_ndim * sizeof(u64));
    break;
  }

  case PLAST_LAYER_BATCHNORM1D: {
    // input: [N, C]
    // y = gamma * (x - mean) / sqrt(var + eps) + beta
    // For simplicity at compile time (inference mode):
    u64 C = l->desc.batchnorm.num_features;

    Tensor *gamma = alloc_tensor(m, (u64[]){C}, 1, true, NULL);
    {
      u64 n = C;
      float *buf = (float *)malloc(n * sizeof(float));
      for (u64 i = 0; i < n; ++i)
        buf[i] = 1.0f;
      memcpy(gamma->data, buf, n * sizeof(float));
      free(buf);
    }
    Tensor *beta = alloc_tensor(m, (u64[]){C}, 1, true, zero_init);
    l->weight = gamma;
    l->bias = beta;
    l->param_start = m->num_params;
    l->num_params = 2;
    register_param(m, gamma, "bn_gamma");
    register_param(m, beta, "bn_beta");

    // mean = mean(x, dim=0)
    u64 mean_shape[] = {1, C};
    Tensor *mean = add_node(m, (Tensor *[]){input_tensor}, 1, mean_shape, 2, MEAN,
                            (KernelParams){0, 1, 0.0f});
    Tensor *diff = add_node(m, (Tensor *[]){input_tensor, mean}, 2, in_shape, in_ndim, SUB,
                            (KernelParams){0, 0, 0.0f});

    // var = mean(diff^2, dim=0)
    Tensor *sq = add_node(m, (Tensor *[]){diff, diff}, 2, in_shape, in_ndim, MUL,
                          (KernelParams){0, 0, 0.0f});
    Tensor *var = add_node(m, (Tensor *[]){sq}, 1, mean_shape, 2, MEAN,
                           (KernelParams){0, 1, 0.0f});
    Tensor *eps_t = add_constant(m, l->desc.batchnorm.eps);
    Tensor *var_eps = add_node(m, (Tensor *[]){var, eps_t}, 2, mean_shape, 2, ADD,
                               (KernelParams){0, 0, 0.0f});
    Tensor *std = add_node(m, (Tensor *[]){var_eps}, 1, mean_shape, 2, LOG,
                           (KernelParams){0, 0, 0.0f});
    Tensor *half = add_constant(m, 0.5f);
    Tensor *log_half = add_node(m, (Tensor *[]){std, half}, 2, mean_shape, 2, MUL,
                                (KernelParams){0, 0, 0.0f});
    Tensor *inv_std = add_node(m, (Tensor *[]){log_half}, 1, mean_shape, 2, EXP,
                               (KernelParams){0, 0, 0.0f});

    Tensor *normed = add_node(m, (Tensor *[]){diff, inv_std}, 2, in_shape, in_ndim, MUL,
                              (KernelParams){0, 0, 0.0f});

    // gamma * normed + beta
    Tensor *g_view = add_node(m, (Tensor *[]){gamma}, 1, (u64[]){1, C}, 2, VIEW,
                              (KernelParams){0, 0, 0.0f});
    Tensor *scaled = add_node(m, (Tensor *[]){normed, g_view}, 2, in_shape, in_ndim, MUL,
                              (KernelParams){0, 0, 0.0f});
    Tensor *b_view = add_node(m, (Tensor *[]){beta}, 1, (u64[]){1, C}, 2, VIEW,
                              (KernelParams){0, 0, 0.0f});
    l->output = add_node(m, (Tensor *[]){scaled, b_view}, 2, in_shape, in_ndim, ADD,
                         (KernelParams){0, 0, 0.0f});

    out_ndim = in_ndim;
    memcpy(out_shape, in_shape, in_ndim * sizeof(u64));
    break;
  }

  case PLAST_LAYER_BATCHNORM2D: {
    // input: [N, C, H, W]
    u64 C = l->desc.batchnorm.num_features;
    u64 N = in_shape[0];
    u64 H = in_shape[2];
    u64 W = in_shape[3];

    Tensor *gamma = alloc_tensor(m, (u64[]){C}, 1, true, NULL);
    {
      u64 n = C;
      float *buf = (float *)malloc(n * sizeof(float));
      for (u64 i = 0; i < n; ++i)
        buf[i] = 1.0f;
      memcpy(gamma->data, buf, n * sizeof(float));
      free(buf);
    }
    Tensor *beta = alloc_tensor(m, (u64[]){C}, 1, true, zero_init);
    l->weight = gamma;
    l->bias = beta;
    l->param_start = m->num_params;
    l->num_params = 2;
    register_param(m, gamma, "bn2d_gamma");
    register_param(m, beta, "bn2d_beta");

    // Permute: [N, C, H, W] → [C, N, H, W]
    Tensor *perm1 = add_node(m, (Tensor *[]){input_tensor}, 1, in_shape, in_ndim, TRANSPOSE,
                             (KernelParams){0, 1, 0.0f});
    u64 flat_shape[] = {C, N * H * W};
    Tensor *flat = add_node(m, (Tensor *[]){perm1}, 1, flat_shape, 2, VIEW,
                            (KernelParams){0, 0, 0.0f});

    u64 mean_shape[] = {C, 1};
    Tensor *mean = add_node(m, (Tensor *[]){flat}, 1, mean_shape, 2, MEAN,
                            (KernelParams){1, 1, 0.0f});
    Tensor *diff = add_node(m, (Tensor *[]){flat, mean}, 2, flat_shape, 2, SUB,
                            (KernelParams){0, 0, 0.0f});
    Tensor *sq = add_node(m, (Tensor *[]){diff, diff}, 2, flat_shape, 2, MUL,
                          (KernelParams){0, 0, 0.0f});
    Tensor *var = add_node(m, (Tensor *[]){sq}, 1, mean_shape, 2, MEAN,
                           (KernelParams){1, 1, 0.0f});
    Tensor *eps_t = add_constant(m, l->desc.batchnorm.eps);
    Tensor *var_eps = add_node(m, (Tensor *[]){var, eps_t}, 2, mean_shape, 2, ADD,
                               (KernelParams){0, 0, 0.0f});
    Tensor *log_var = add_node(m, (Tensor *[]){var_eps}, 1, mean_shape, 2, LOG,
                               (KernelParams){0, 0, 0.0f});
    Tensor *half = add_constant(m, 0.5f);
    Tensor *log_half = add_node(m, (Tensor *[]){log_var, half}, 2, mean_shape, 2, MUL,
                                (KernelParams){0, 0, 0.0f});
    Tensor *inv_std = add_node(m, (Tensor *[]){log_half}, 1, mean_shape, 2, EXP,
                               (KernelParams){0, 0, 0.0f});

    Tensor *normed_flat = add_node(m, (Tensor *[]){diff, inv_std}, 2, flat_shape, 2, MUL,
                                   (KernelParams){0, 0, 0.0f});

    u64 perm_back_shape[] = {C, N, H, W};
    Tensor *normed_perm = add_node(m, (Tensor *[]){normed_flat}, 1, perm_back_shape, 4, VIEW,
                                   (KernelParams){0, 0, 0.0f});
    Tensor *normed = add_node(m, (Tensor *[]){normed_perm}, 1, in_shape, in_ndim, TRANSPOSE,
                              (KernelParams){0, 1, 0.0f});

    Tensor *g_view = add_node(m, (Tensor *[]){gamma}, 1, (u64[]){1, C, 1, 1}, 4, VIEW,
                              (KernelParams){0, 0, 0.0f});
    Tensor *scaled = add_node(m, (Tensor *[]){normed, g_view}, 2, in_shape, in_ndim, MUL,
                              (KernelParams){0, 0, 0.0f});
    Tensor *b_view = add_node(m, (Tensor *[]){beta}, 1, (u64[]){1, C, 1, 1}, 4, VIEW,
                              (KernelParams){0, 0, 0.0f});
    l->output = add_node(m, (Tensor *[]){scaled, b_view}, 2, in_shape, in_ndim, ADD,
                         (KernelParams){0, 0, 0.0f});

    out_ndim = in_ndim;
    memcpy(out_shape, in_shape, in_ndim * sizeof(u64));
    break;
  }

  case PLAST_LAYER_LAYERNORM: {
    // layer_norm(x) = gamma * (x - mean) / sqrt(var + eps) + beta
    // mean, var over last normalized_shape dims
    u64 total_flat = product(in_shape, in_ndim);
    u64 norm_dim = l->desc.layernorm.normalized_shape;
    u64 batch = total_flat / norm_dim;

    u64 flat_shape_norm[] = {batch, norm_dim};
    Tensor *flat_n = add_node(m, (Tensor *[]){input_tensor}, 1, flat_shape_norm, 2, VIEW,
                              (KernelParams){0, 0, 0.0f});

    u64 mean_shape_n[] = {batch, 1};
    Tensor *mean_n = add_node(m, (Tensor *[]){flat_n}, 1, mean_shape_n, 2, MEAN,
                              (KernelParams){1, 1, 0.0f});
    Tensor *diff_n = add_node(m, (Tensor *[]){flat_n, mean_n}, 2, flat_shape_norm, 2, SUB,
                              (KernelParams){0, 0, 0.0f});
    Tensor *sq_n = add_node(m, (Tensor *[]){diff_n, diff_n}, 2, flat_shape_norm, 2, MUL,
                            (KernelParams){0, 0, 0.0f});
    Tensor *var_n = add_node(m, (Tensor *[]){sq_n}, 1, mean_shape_n, 2, MEAN,
                             (KernelParams){1, 1, 0.0f});
    Tensor *eps_t = add_constant(m, l->desc.layernorm.eps);
    Tensor *var_eps_n = add_node(m, (Tensor *[]){var_n, eps_t}, 2, mean_shape_n, 2, ADD,
                                 (KernelParams){0, 0, 0.0f});
    Tensor *log_var_n = add_node(m, (Tensor *[]){var_eps_n}, 1, mean_shape_n, 2, LOG,
                                 (KernelParams){0, 0, 0.0f});
    Tensor *half = add_constant(m, 0.5f);
    Tensor *log_half_n = add_node(m, (Tensor *[]){log_var_n, half}, 2, mean_shape_n, 2, MUL,
                                  (KernelParams){0, 0, 0.0f});
    Tensor *inv_std_n = add_node(m, (Tensor *[]){log_half_n}, 1, mean_shape_n, 2, EXP,
                                 (KernelParams){0, 0, 0.0f});

    Tensor *normed_n = add_node(m, (Tensor *[]){diff_n, inv_std_n}, 2, flat_shape_norm, 2, MUL,
                                (KernelParams){0, 0, 0.0f});

    Tensor *gamma_n = alloc_tensor(m, (u64[]){norm_dim}, 1, true, NULL);
    {
      float *buf = (float *)malloc(norm_dim * sizeof(float));
      for (u64 i = 0; i < norm_dim; ++i)
        buf[i] = 1.0f;
      memcpy(gamma_n->data, buf, norm_dim * sizeof(float));
      free(buf);
    }
    Tensor *beta_n = alloc_tensor(m, (u64[]){norm_dim}, 1, true, zero_init);
    l->weight = gamma_n;
    l->bias = beta_n;
    l->param_start = m->num_params;
    l->num_params = 2;
    register_param(m, gamma_n, "ln_gamma");
    register_param(m, beta_n, "ln_beta");

    Tensor *scaled_n = add_node(m, (Tensor *[]){normed_n, gamma_n}, 2, flat_shape_norm, 2, MUL,
                                (KernelParams){0, 0, 0.0f});
    Tensor *output_n = add_node(m, (Tensor *[]){scaled_n, beta_n}, 2, flat_shape_norm, 2, ADD,
                                (KernelParams){0, 0, 0.0f});

    l->output = add_node(m, (Tensor *[]){output_n}, 1, in_shape, in_ndim, VIEW,
                         (KernelParams){0, 0, 0.0f});

    out_ndim = in_ndim;
    memcpy(out_shape, in_shape, in_ndim * sizeof(u64));
    break;
  }
  }

  l->output_ndim = out_ndim;
  memcpy(l->output_shape, out_shape, out_ndim * sizeof(u64));
}

void plast_model_compile(PlastModel *m, const u64 *input_shape, u64 input_ndim) {
  if (m->compiled)
    return;

  srand(42);
  m->meta = arena_create(META_SIZE, CPU);
  m->data = arena_create(DATA_SIZE, m->device);

  m->input_ndim = input_ndim;
  memcpy(m->input_shape, input_shape, input_ndim * sizeof(u64));

  m->input_tensor = alloc_tensor(m, input_shape, input_ndim, false, NULL);

  Tensor *current = m->input_tensor;
  for (int i = 0; i < m->num_layers; ++i) {
    compile_layer(m, i, current->shape, current->ndim, current);
    current = m->layers[i].output;
  }

  m->output_tensor = current;
  m->compiled = true;

  JIT *jit = jit_create(16);
  m->scheduler = init_scheduler(jit);
}

// --- Forward / Backward ---

static void zero_dag_grads(DAG *dag) {
  for (u32 i = 0; i < dag->count; ++i) {
    Node *n = dag->nodes[i];
    if (n->output && n->output->grad)
      zeros(n->output->grad, numel(n->output->grad));
    for (int j = 0; j < n->num_inputs; ++j) {
      if (n->inputs[j] && n->inputs[j]->grad)
        zeros(n->inputs[j]->grad, numel(n->inputs[j]->grad));
    }
  }
}

static void zero_dag_data(DAG *dag) {
  for (u32 i = 0; i < dag->count; ++i) {
    Node *n = dag->nodes[i];
    if (n->output)
      zeros(n->output, numel(n->output));
  }
}

Tensor *plast_model_forward(PlastModel *m) {
  if (!m->compiled || !m->output_tensor || !m->output_tensor->creator)
    return m->output_tensor;

  DAG *dag = alloc_dag(MIN_DAG_CAPACITY);
  build_dag(dag, m->output_tensor->creator);
  zero_dag_data(dag);
  dag_forward(dag);
  dag_release(dag);

  return m->output_tensor;
}

void plast_model_zero_grad(PlastModel *m) {
  for (int i = 0; i < m->num_params; ++i) {
#if defined(CUDA_AVAILABLE)
    if (m->params[i]->device == CUDA) {
      zero_grad_cuda(m->params[i]);
    } else {
      zero_grad_cpu(m->params[i]);
    }
#else
    zero_grad_cpu(m->params[i]);
#endif
  }
}

static void force_reset_flags(Node *node) {
  if (!node)
    return;
  node->visited = false;
  for (u64 i = 0; i < node->num_inputs; ++i) {
    if (node->inputs[i]->creator)
      force_reset_flags(node->inputs[i]->creator);
  }
}

void plast_model_backward(PlastModel *m, Tensor *loss) {
  if (!loss || !loss->creator)
    return;

  // Reset visited flags across entire graph (model + loss)
  // build_dag's reset_node_flags only resets nodes with visited=true, but
  // loss nodes are fresh (visited=false) while model nodes are visited=true
  // from the forward DAG, so we need a forced reset of everything.
  force_reset_flags(loss->creator);

  // Build full graph from loss root (includes model + loss nodes)
  DAG *dag = alloc_dag(MIN_DAG_CAPACITY);
  build_dag(dag, loss->creator);

  // Zero all grads and data before computing
  zero_dag_grads(dag);
  zero_dag_data(dag);

  // Forward: compute output + loss
  dag_forward(dag);

  // Backward: propagate gradients
  set_ones_grad(loss);
  dag_backward(dag);

  dag_release(dag);
}

void plast_optimizer_step(PlastModel *m) {
  if (!m->optimizer || m->num_params == 0)
    return;
  m->optimizer->step_fn(m->optimizer->state, m->params, m->num_params);
}

Tensor *plast_model_input(PlastModel *m) {
  return m->input_tensor;
}

Tensor *plast_model_output(PlastModel *m) {
  return m->output_tensor;
}

Arena *plast_model_meta_arena(PlastModel *m) {
  return &m->meta;
}

Arena *plast_model_data_arena(PlastModel *m) {
  return &m->data;
}

void plast_model_set_optimizer(PlastModel *m, PlastOptimizer *opt) {
  if (m->optimizer)
    plast_optim_free(m->optimizer);
  m->optimizer = opt;
}

// --- Utility functions ---

void plast_model_summary(const PlastModel *m) {
  printf("PlastModel\n");
  printf("├── Device: %s\n", m->device == CUDA ? "CUDA" : "CPU");
  printf("├── Compiled: %s\n", m->compiled ? "yes" : "no");
  printf("├── Input: [");
  for (u64 i = 0; i < m->input_ndim; ++i) {
    printf("%lu", m->input_shape[i]);
    if (i < m->input_ndim - 1)
      printf(", ");
  }
  printf("]\n");
  printf("├── Layers: %d\n", m->num_layers);
  for (int i = 0; i < m->num_layers; ++i) {
    const PlastLayer *l = &m->layers[i];
    printf("│   %d: ", i);
    switch (l->desc.type) {
    case PLAST_LAYER_DENSE:
      printf("Dense(%lu -> %lu, bias=%s)", l->output_shape[1], l->desc.dense.out_features,
             l->desc.dense.bias ? "true" : "false");
      break;
    case PLAST_LAYER_CONV2D:
      printf("Conv2d(%lu out, kernel=%lu, stride=%lu, bias=%s)", l->desc.conv2d.out_channels,
             l->desc.conv2d.kernel_size, l->desc.conv2d.stride,
             l->desc.conv2d.bias ? "true" : "false");
      break;
    case PLAST_LAYER_ACTIVATION: {
      const char *names[] = {"None", "ReLU", "LeakyReLU", "Sigmoid", "Tanh", "Softmax"};
      printf("Activation(%s)", names[l->desc.activation.act]);
      break;
    }
    case PLAST_LAYER_FLATTEN:
      printf("Flatten");
      break;
    case PLAST_LAYER_DROPOUT:
      printf("Dropout(p=%.2f)", l->desc.dropout.p);
      break;
    case PLAST_LAYER_BATCHNORM1D:
      printf("BatchNorm1d(%lu)", l->desc.batchnorm.num_features);
      break;
    case PLAST_LAYER_BATCHNORM2D:
      printf("BatchNorm2d(%lu)", l->desc.batchnorm.num_features);
      break;
    case PLAST_LAYER_LAYERNORM:
      printf("LayerNorm(%lu)", l->desc.layernorm.normalized_shape);
      break;
    }
    printf(" → [");
    for (u64 j = 0; j < l->output_ndim; ++j) {
      printf("%lu", l->output_shape[j]);
      if (j < l->output_ndim - 1)
        printf(", ");
    }
    printf("]\n");
  }
  printf("├── Parameters: %d\n", m->num_params);
  u64 total = 0;
  for (int i = 0; i < m->num_params; ++i) {
    total += numel(m->params[i]);
  }
  printf("└── Total params: %lu\n", total);
}

void plast_model_print_weights(const PlastModel *m) {
  printf("Model weights:\n");
  for (int i = 0; i < m->num_params; ++i) {
    const Tensor *t = m->params[i];
    u64 n = numel(t);
    float *buf = (float *)malloc(n * sizeof(float));
    memcpy(buf, t->data, n * sizeof(float));
    printf("  %s [", m->param_names[i]);
    for (u64 j = 0; j < t->ndim; ++j) {
      printf("%lu", t->shape[j]);
      if (j < t->ndim - 1)
        printf("x");
    }
    printf("]: ");
    for (u64 j = 0; j < n && j < 8; ++j)
      printf("%.4f ", buf[j]);
    if (n > 8)
      printf("...");
    printf("\n");
    free(buf);
  }
}

void plast_model_set_weight_init(PlastModel *m, int layer_idx, plast_init_fn fn) {
  if (layer_idx < 0 || layer_idx >= m->num_layers)
    return;
  PlastLayer *l = &m->layers[layer_idx];
  if (l->weight && fn)
    fn(l->weight, numel(l->weight));
}

// --- Tensor utilities ---

Tensor *plast_tensor_from_array(const float *data, const u64 *shape, u64 ndim) {
  u64 *strides = compute_strides(shape, ndim);
  u64 n = product(shape, ndim);
  Tensor *t = (Tensor *)malloc(sizeof(Tensor));
  memset(t, 0, sizeof(Tensor));
  t->ndim = ndim;
  t->device = CPU;
  t->dtype = FLOAT32;
  t->requires_grad = false;
  t->grad = NULL;
  t->creator = NULL;
  memcpy(t->shape, shape, ndim * sizeof(u64));
  memcpy(t->strides, strides, ndim * sizeof(u64));
  t->data = malloc(n * sizeof(float));
  memcpy(t->data, data, n * sizeof(float));
  free(strides);
  return t;
}

Tensor *plast_scalar(float value, DEVICE device) {
  Tensor *t = (Tensor *)malloc(sizeof(Tensor));
  memset(t, 0, sizeof(Tensor));
  t->ndim = 1;
  t->shape[0] = 1;
  t->strides[0] = 1;
  t->device = device;
  t->dtype = FLOAT32;
  t->requires_grad = false;
  t->grad = NULL;
  t->creator = NULL;
  t->data = malloc(sizeof(float));
  *(float *)t->data = value;
  return t;
}

void plast_tensor_print(const Tensor *t, const char *name) {
  u64 n = numel(t);
  float *buf = (float *)malloc(n * sizeof(float));
  memcpy(buf, t->data, n * sizeof(float));

  printf("%s [", name ? name : "Tensor");
  for (u64 i = 0; i < t->ndim; ++i) {
    printf("%lu", t->shape[i]);
    if (i < t->ndim - 1)
      printf("x");
  }
  printf("]:\n[");

  for (u64 i = 0; i < n; ++i) {
    printf(" %.4f", buf[i]);
    if ((i + 1) % t->shape[t->ndim - 1] == 0 && i < n - 1)
      printf("\n ");
  }
  printf(" ]\n");
  free(buf);
}
