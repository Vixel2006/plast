#pragma once

#include "op.h"

typedef struct Node {
  Tensor **inputs;
  Tensor *output;
  Op op;
  int num_inputs;
  bool visited;
  bool on_stack;
  u64 dim;
  u64 keepdim;
} Node;

Node *arena_node_alloc(Arena *a, Tensor **inputs, int num_inputs,
                       Tensor *output, Op op, u64 dim, u64 keepdim);
void execute_forward(Node *n);
void execute_backward(Node *n);
