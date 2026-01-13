#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "graph.h"
#include "kernels/add.h"
#include "kernels/cpu_utils.h"
#include "kernels/div.h"
#include "kernels/exp.h"
#include "kernels/leaky_relu.h"
#include "kernels/matmul.h"
#include "kernels/mean.h"
#include "kernels/mul.h"
#include "kernels/neg.h"
#include "kernels/sub.h"
#include "kernels/tan.h"
#include "node.h"
#include "op.h"
#include "optimizers/sgd.h"
#include "optimizers/zero_grad.h"
#include "tensor.h"

// Helper to print a tensor
void print_tensor(Tensor *t, const char *name) {
  printf("%s (shape: [", name);
  for (u64 i = 0; i < t->ndim; ++i) {
    printf("%lu", t->shape[i]);
    if (i < t->ndim - 1)
      printf(", ");
  }
  printf("]):\n");

  u64 total_elements = numel(t);
  if (total_elements == 0) {
    printf("[]\n\n");
    return;
  }
  float *data_h = (float *)malloc(total_elements * sizeof(float));

  if (t->device == CUDA) {
    printf("Cannot print CUDA tensor data from host.\n");
    free(data_h);
    return;
  } else {
    memcpy(data_h, t->data, total_elements * sizeof(float));
  }

  printf("[");
  for (u64 i = 0; i < total_elements; ++i) {
    printf("%.4f", data_h[i]);
    if (i < total_elements - 1)
      printf(", ");
  }
  printf("]\n\n");
  free(data_h);
}

int main() {
  Arena a = arena_create(Mib(3), CPU);
  Arena ac = arena_create(Mib(3), CPU);

  Tensor *t1 = init(&a, &ac, CPU, FLOAT32, (u64[]){2, 2}, 2, true, ones);
  Tensor *t2 = init(&a, &ac, CPU, FLOAT32, (u64[]){2, 2}, 2, true, ones);
  Tensor *t3 = arena_tensor_alloc(&a, &ac, (u64[]){2, 2}, 2, (u64[]){2, 1},
                                  FLOAT32, true, NULL, CPU);
  Tensor *t4 = init(&a, &ac, CPU, FLOAT32, (u64[]){2, 2}, 2, true, ones);
  Tensor *t5 = arena_tensor_alloc(&a, &ac, (u64[]){2, 2}, 2, (u64[]){2, 1},
                                  FLOAT32, true, NULL, CPU);

  zeros(t3, 4);
  zeros(t5, 4);
  set_ones_grad(t5);

  Node *node1 = arena_node_alloc(&a, (Tensor *[]){t1, t2}, 2, t3,
                                 get_op_impl(MATMUL), 0, false);
  Node *node2 = arena_node_alloc(&a, (Tensor *[]){t3, t4}, 2, t5,
                                 get_op_impl(ADD), 0, false);

  printf("node 1 = %p, node 2 = %p\n", t3->creator, t5->creator);

  forward(node2);
  backward(node2);
  //  backward(node1);

  print_tensor(t5, "t5");
  print_tensor(t4, "t4");
  print_tensor(t3, "t3");
  print_tensor(t2, "t2");
  print_tensor(t1, "t1");
  print_tensor(t1->grad, "grad t1");
  print_tensor(t2->grad, "grad t2");
  print_tensor(t3->grad, "grad t3");
  print_tensor(t4->grad, "grad t4");
  print_tensor(t5->grad, "grad t5");

  arena_release(&a);
  arena_release(&ac);

  return 0;
}

/*
// Initialization function for random weights
void random_uniform_init(Tensor *t, u64 num_elements) {
  float *data = (float *)t->data;
  for (u64 i = 0; i < num_elements; ++i) {
    data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // Range [-1, 1]
  }
}

// --- Operation Wrappers ---
// These wrappers simplify the process of creating computation graph nodes.

Tensor *_apply_op(Op op, Tensor **inputs, int num_inputs, u64 *output_shape,
                  u64 output_ndim, u64 dim, bool keepdim) {
  Tensor *output = init(&meta_arena, &compute_arena, CPU, FLOAT32, output_shape,
                        output_ndim, true, zeros);
  Node *node = arena_node_alloc(&meta_arena, inputs, num_inputs, output, op,
                                dim, keepdim);
  output->creator = node;
  return output;
}

Tensor *_matmul(Tensor *a, Tensor *b) {
  u64 output_shape[] = {a->shape[0], b->shape[1]};
  Op op = get_op_impl(MATMUL);
  return _apply_op(op, (Tensor *[]){a, b}, 2, output_shape, 2, 0, false);
}

Tensor *_add(Tensor *a, Tensor *b) {
  u64 broadcast_shape[MAX_NDIM];
  u64 broadcast_ndim;
  get_broadcast_shape(a->shape, a->ndim, b->shape, b->ndim, broadcast_shape,
                      &broadcast_ndim);
  Op op = get_op_impl(ADD);
  return _apply_op(op, (Tensor *[]){a, b}, 2, broadcast_shape, broadcast_ndim,
                   0, false);
}

Tensor *_sub(Tensor *a, Tensor *b) {
  u64 broadcast_shape[MAX_NDIM];
  u64 broadcast_ndim;
  get_broadcast_shape(a->shape, a->ndim, b->shape, b->ndim, broadcast_shape,
                      &broadcast_ndim);
  Op op = get_op_impl(SUB);
  return _apply_op(op, (Tensor *[]){a, b}, 2, broadcast_shape, broadcast_ndim,
                   0, false);
}

Tensor *_mul(Tensor *a, Tensor *b) {
  u64 broadcast_shape[MAX_NDIM];
  u64 broadcast_ndim;
  get_broadcast_shape(a->shape, a->ndim, b->shape, b->ndim, broadcast_shape,
                      &broadcast_ndim);
  Op op = get_op_impl(MUL);
  return _apply_op(op, (Tensor *[]){a, b}, 2, broadcast_shape, broadcast_ndim,
                   0, false);
}

Tensor *_leaky_relu(Tensor *a) {
  Op op = get_op_impl(LEAKY_RELU);
  return _apply_op(op, (Tensor *[]){a}, 1, a->shape, a->ndim, 0, false);
}

Tensor *_mean(Tensor *a) {
  u64 output_shape[] = {1};
  Op op = get_op_impl(MEAN);
  return _apply_op(op, (Tensor *[]){a}, 1, output_shape, 1, 0, false);
}

// --- Neural Network Layer Definition ---

typedef struct {
  Tensor *W;
  Tensor *b;
} Linear;

Linear init_linear(u64 in_features, u64 out_features, bool requires_grad) {
  Linear layer;
  u64 w_shape[] = {in_features, out_features};
  layer.W = init(&meta_arena, &data_arena, CPU, FLOAT32, w_shape, 2,
                 requires_grad, random_uniform_init);

  u64 b_shape[] = {1, out_features};
  layer.b = init(&meta_arena, &data_arena, CPU, FLOAT32, b_shape, 2,
                 requires_grad, zeros);
  return layer;
}

Tensor *linear_forward(Linear *layer, Tensor *x) {
  Tensor *matmul_res = _matmul(x, layer->W);
  Tensor *add_res = _add(matmul_res, layer->b);
  return add_res;
}

// --- Main Training ---

int main() {
  srand(time(NULL));

  // Initialize arenas
  meta_arena = arena_create(Mib(64), CPU);
  data_arena = arena_create(Mib(64), CPU);
  compute_arena = arena_create(Mib(64), CPU);

  printf("Training a 2-layer neural network for the XOR problem.\n");
  printf("A 1-layer network cannot solve XOR, so a hidden layer is used.\n\n");

  // 1. Prepare Data
  u64 x_shape[] = {1, 2};
  Tensor *X_train =
      init(&meta_arena, &data_arena, CPU, FLOAT32, x_shape, 2, false, zeros);
  float x_data[] = {0.0f, 1.0f};
  memcpy(X_train->data, x_data, sizeof(x_data));

  u64 y_shape[] = {1, 1};
  Tensor *Y_train =
      init(&meta_arena, &data_arena, CPU, FLOAT32, y_shape, 2, false, zeros);
  float y_data[] = {1.0f};
  memcpy(Y_train->data, y_data, sizeof(y_data));

  print_tensor(X_train, "X_train");
  print_tensor(Y_train, "Y_train");

  // 2. Define Model
  Linear layer1 = init_linear(2, 2, true); // Hidden layer: 2 inputs, 2 outputs
  Linear layer2 = init_linear(2, 1, true); // Output layer: 2 inputs, 1 output

  Tensor *params[] = {layer1.W, layer1.b, layer2.W, layer2.b};
  int num_params = sizeof(params) / sizeof(params[0]);

  // 3. Optimizer
  float learning_rate = 0.1f;
  SGD sgd = arena_alloc_sgd(learning_rate);

  // 4. Training Loop
  int epochs = 2000;
  printf("Starting training for %d epochs with LR=%.2f...\n\n", epochs,
         learning_rate);

  for (int i = 0; i < epochs; ++i) {
    // --- Forward pass ---
    Tensor *h = linear_forward(&layer1, X_train);
    Tensor *h_act = _leaky_relu(h);
    Tensor *y_logits = linear_forward(&layer2, h_act);
    Tensor *y_pred = _leaky_relu(y_logits);

    // --- Compute loss (MSE) ---
    Tensor *diff = _sub(y_pred, Y_train);
    Tensor *sq_diff = _mul(diff, diff);
    Tensor *loss = _mean(sq_diff);

    if (i % 200 == 0 || i == epochs - 1) {
      print_tensor(loss, "Loss");
    }

    // --- Backward pass ---
    set_ones_grad(loss);
    backward(loss->creator); // Start backward from the loss node

    // --- Update weights ---
    sgd_step_cpu(&sgd, params, num_params);

    // --- Zero gradients for next iteration ---
    for (int j = 0; j < num_params; ++j) {
      zero_grad_cpu(params[j]);
    }

    // Reset the computation arena for the next forward pass
    arena_reset(&compute_arena);
  }

  printf("\n--- Training Finished ---\n\n");

  // 5. Final Predictions
  Tensor *h = linear_forward(&layer1, X_train);
  Tensor *h_act = _leaky_relu(h);
  Tensor *y_logits = linear_forward(&layer2, h_act);
  Tensor *y_pred = _leaky_relu(y_logits);

  print_tensor(y_pred, "Final Predictions");
  print_tensor(Y_train, "True Labels");

  // Release memory
  arena_release(&compute_arena);
  arena_release(&data_arena);
  arena_release(&meta_arena);

  return 0;
}
*/
