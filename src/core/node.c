#include "core/node.h"

Node *arena_node_alloc(Arena *a, Tensor **inputs, int num_inputs, Tensor *output, Op op,
                       OP_TYPE op_type, KernelParams params) {
  Node *node = (Node *)arena_alloc(a, sizeof(Node), 8);

  node->inputs = inputs;
  node->output = output;
  node->op = op;
  node->op_type = op_type;

  node->output->creator = node;

  node->num_inputs = num_inputs;
  node->visited = false;
  node->on_stack = false;
  node->params = params;

  return node;
}

void execute_forward(Node *n) {
  ForwardKernel forward_kernel = forward_kernel_dispatcher(n->op, n->output->device);

  forward_kernel((const Tensor **)n->inputs, n->output, n->params);
}

void execute_backward(Node *n) {
  BackwardKernel backward_kernel = backward_kernel_dispatcher(n->op, n->output->device);

  backward_kernel(n->inputs, n->output, n->params);
}
