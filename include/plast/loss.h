#pragma once

#include "core/tensor.h"
#include "core/graph.h"
#include "core/op.h"
#include "core/node.h"
#include "core/arena.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  Tensor *output;
  Node *last_node;
} PlastLoss;

PlastLoss plast_mse_loss(Tensor *pred, Tensor *target, Arena *meta, Arena *data);
PlastLoss plast_l1_loss(Tensor *pred, Tensor *target, Arena *meta, Arena *data);
PlastLoss plast_cross_entropy_loss(Tensor *pred, Tensor *target, Arena *meta, Arena *data);

#ifdef __cplusplus
}
#endif
