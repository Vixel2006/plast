#include "plast/loss.h"
#include "core/arena.h"
#include "core/node.h"
#include "core/op.h"
#include "core/tensor.h"
#include <string.h>

static Tensor *alloc_loss_tensor(Arena *meta, Arena *data, DEVICE device, const u64 *shape,
                                 u64 ndim) {
  u64 *strides = (u64 *)arena_alloc(meta, ndim * sizeof(u64), 8);
  if (!strides)
    return NULL;
  strides[ndim - 1] = 1;
  for (u64 i = ndim - 1; i > 0; --i)
    strides[i - 1] = strides[i] * shape[i];

  Tensor *t = arena_tensor_alloc(meta, data, (u64 *)shape, ndim, strides, FLOAT32, true, NULL,
                                 device);
  return t;
}

static Tensor *add_loss_node(Arena *meta, Arena *data, Tensor **inputs, int num_inputs,
                             const u64 *shape, u64 ndim, OP_TYPE op_type, KernelParams params,
                             DEVICE device) {
  Tensor *output = alloc_loss_tensor(meta, data, device, shape, ndim);
  if (!output)
    return NULL;
  Tensor **inputs_copy = (Tensor **)arena_alloc(meta, num_inputs * sizeof(Tensor *), 8);
  for (int i = 0; i < num_inputs; ++i)
    inputs_copy[i] = inputs[i];
  arena_node_alloc(meta, inputs_copy, num_inputs, output, get_op_impl(op_type), op_type, params);
  return output;
}

PlastLoss plast_mse_loss(Tensor *pred, Tensor *target, Arena *meta, Arena *data) {
  PlastLoss loss = {NULL, NULL};

  u64 out_ndim = pred->ndim;
  u64 out_shape[MAX_NDIM];
  memcpy(out_shape, pred->shape, out_ndim * sizeof(u64));

  // diff = pred - target
  Tensor *diff =
      add_loss_node(meta, data, (Tensor *[]){pred, target}, 2, out_shape, out_ndim, SUB,
                    (KernelParams){0, 0, 0.0f}, pred->device);
  if (!diff)
    return loss;

  // sq_diff = diff * diff
  Tensor *sq_diff =
      add_loss_node(meta, data, (Tensor *[]){diff, diff}, 2, out_shape, out_ndim, MUL,
                    (KernelParams){0, 0, 0.0f}, pred->device);
  if (!sq_diff)
    return loss;

  // mean over all dims
  u64 scalar_shape[] = {1};
  Tensor *mean =
      add_loss_node(meta, data, (Tensor *[]){sq_diff}, 1, scalar_shape, 1, MEAN,
                    (KernelParams){MAX_NDIM + 1, 0, 0.0f}, pred->device);

  loss.output = mean;
  loss.last_node = mean ? mean->creator : NULL;
  return loss;
}

PlastLoss plast_l1_loss(Tensor *pred, Tensor *target, Arena *meta, Arena *data) {
  PlastLoss loss = {NULL, NULL};

  u64 out_ndim = pred->ndim;
  u64 out_shape[MAX_NDIM];
  memcpy(out_shape, pred->shape, out_ndim * sizeof(u64));

  // diff = pred - target
  Tensor *diff =
      add_loss_node(meta, data, (Tensor *[]){pred, target}, 2, out_shape, out_ndim, SUB,
                    (KernelParams){0, 0, 0.0f}, pred->device);
  if (!diff)
    return loss;

  // abs_diff = |diff|
  Tensor *abs_diff = add_loss_node(meta, data, (Tensor *[]){diff}, 1, out_shape, out_ndim, ABS,
                                   (KernelParams){0, 0, 0.0f}, pred->device);
  if (!abs_diff)
    return loss;

  u64 scalar_shape[] = {1};
  Tensor *mean =
      add_loss_node(meta, data, (Tensor *[]){abs_diff}, 1, scalar_shape, 1, MEAN,
                    (KernelParams){MAX_NDIM + 1, 0, 0.0f}, pred->device);

  loss.output = mean;
  loss.last_node = mean ? mean->creator : NULL;
  return loss;
}

PlastLoss plast_cross_entropy_loss(Tensor *pred, Tensor *target, Arena *meta, Arena *data) {
  PlastLoss loss = {NULL, NULL};
  DEVICE device = pred->device;

  u64 N = pred->shape[0];
  u64 C = pred->shape[1];

  // log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
  u64 max_shape[] = {N, 1};
  Tensor *max_val =
      add_loss_node(meta, data, (Tensor *[]){pred}, 1, max_shape, 2, MAX,
                    (KernelParams){1, 1, 0.0f}, device);
  if (!max_val)
    return loss;

  u64 full_shape[] = {N, C};
  Tensor *shifted =
      add_loss_node(meta, data, (Tensor *[]){pred, max_val}, 2, full_shape, 2, SUB,
                    (KernelParams){0, 0, 0.0f}, device);
  if (!shifted)
    return loss;

  Tensor *exp = add_loss_node(meta, data, (Tensor *[]){shifted}, 1, full_shape, 2, EXP,
                              (KernelParams){0, 0, 0.0f}, device);
  if (!exp)
    return loss;

  Tensor *sum_exp =
      add_loss_node(meta, data, (Tensor *[]){exp}, 1, max_shape, 2, SUM,
                    (KernelParams){1, 1, 0.0f}, device);
  if (!sum_exp)
    return loss;

  Tensor *log_sum =
      add_loss_node(meta, data, (Tensor *[]){sum_exp}, 1, max_shape, 2, LOG,
                    (KernelParams){0, 0, 0.0f}, device);
  if (!log_sum)
    return loss;

  Tensor *log_soft =
      add_loss_node(meta, data, (Tensor *[]){shifted, log_sum}, 2, full_shape, 2, SUB,
                    (KernelParams){0, 0, 0.0f}, device);
  if (!log_soft)
    return loss;

  // nll_loss = -mean(target * log_soft)
  Tensor *target_log =
      add_loss_node(meta, data, (Tensor *[]){target, log_soft}, 2, full_shape, 2, MUL,
                    (KernelParams){0, 0, 0.0f}, device);
  if (!target_log)
    return loss;

  Tensor *sum =
      add_loss_node(meta, data, (Tensor *[]){target_log}, 1, max_shape, 2, SUM,
                    (KernelParams){1, 1, 0.0f}, device);
  if (!sum)
    return loss;

  Tensor *neg =
      add_loss_node(meta, data, (Tensor *[]){sum}, 1, max_shape, 2, NEG,
                    (KernelParams){0, 0, 0.0f}, device);
  if (!neg)
    return loss;

  u64 scalar_shape[] = {1};
  Tensor *mean =
      add_loss_node(meta, data, (Tensor *[]){neg}, 1, scalar_shape, 1, MEAN,
                    (KernelParams){MAX_NDIM + 1, 0, 0.0f}, device);

  loss.output = mean;
  loss.last_node = mean ? mean->creator : NULL;
  return loss;
}
