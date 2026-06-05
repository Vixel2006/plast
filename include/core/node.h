#pragma once

#include "core/op.h"

typedef struct Node {
  Tensor **inputs;
  Tensor *output;
  Op op;
  OP_TYPE op_type;
  int num_inputs;
  bool visited;
  bool on_stack;
  KernelParams params;
} Node;

Node *arena_node_alloc(Arena *a, Tensor **inputs, int num_inputs, Tensor *output, Op op,
                       OP_TYPE op_type, KernelParams params);
void execute_forward(Node *n);
void execute_backward(Node *n);
