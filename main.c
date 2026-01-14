#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "graph.h"
#include "kernels/abs.h"
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

void rand_init(Tensor *t, u64 num_elements) {
  float scale = sqrtf(2.0f / (float)t->shape[0]);
  float *host_buffer = (float *)malloc(num_elements * sizeof(float));
  for (u64 i = 0; i < num_elements; ++i) {
    host_buffer[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
  }

  if (t->device == CUDA) {
    // We need an arena to use arena_memcpy_h2d, or we can use cudaMemcpy directly if we include it.
    // However, the library seems to prefer using the arena abstraction.
    // But rand_init doesn't have access to the arena.
    // Let's use cudaMemcpy if we can, or modify the library.
    // Looking at arena_cuda.cu, it defines extern "C" functions.
    extern void arena_memcpy_h2d_cuda(void *dest, const void *src, u64 size);
    arena_memcpy_h2d_cuda(t->data, host_buffer, num_elements * sizeof(float));
  } else {
    memcpy(t->data, host_buffer, num_elements * sizeof(float));
  }
  free(host_buffer);
}

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
    extern void arena_memcpy_d2h_cuda(void *dest, const void *src, u64 size);
    arena_memcpy_d2h_cuda(data_h, t->data, total_elements * sizeof(float));
  } else {
    memcpy(data_h, t->data, total_elements * sizeof(float));
  }

  printf("[");
  for (u64 i = 0; i < total_elements; ++i) {
    printf("%.4f", data_h[i]);
    if (i < total_elements - 1)
      printf(", ");
    if ((i + 1) % t->shape[t->ndim - 1] == 0 && i < total_elements - 1)
        printf("\n ");
  }
  printf("]\n\n");
  free(data_h);
}

void copy_h2d(Tensor *t, const void *src) {
    if (t->device == CUDA) {
        extern void arena_memcpy_h2d_cuda(void *dest, const void *src, u64 size);
        arena_memcpy_h2d_cuda(t->data, src, numel(t) * dtype_size(t->dtype));
    } else {
        memcpy(t->data, src, numel(t) * dtype_size(t->dtype));
    }
}

int main() {
  srand(42);
  DEVICE device = CUDA; // Switch to CUDA
  Arena a = arena_create(Mib(10), CPU); // Metadata MUST be on CPU
  Arena ac = arena_create(Mib(100), device);

  // XOR Data
  float x_data[] = {0, 0, 0, 1, 1, 0, 1, 1};
  float y_data[] = {0, 1, 1, 0};

  Tensor *X = init(&a, &ac, device, FLOAT32, (u64[]){4, 2}, 2, false, NULL);
  copy_h2d(X, x_data);

  Tensor *Y = init(&a, &ac, device, FLOAT32, (u64[]){4, 1}, 2, false, NULL);
  copy_h2d(Y, y_data);

  // Parameters
  int hidden_size = 8;
  Tensor *W1 = init(&a, &ac, device, FLOAT32, (u64[]){2, hidden_size}, 2, true, rand_init);
  Tensor *b1 = init(&a, &ac, device, FLOAT32, (u64[]){1, hidden_size}, 2, true, zeros);
  Tensor *W2 = init(&a, &ac, device, FLOAT32, (u64[]){hidden_size, 1}, 2, true, rand_init);
  Tensor *b2 = init(&a, &ac, device, FLOAT32, (u64[]){1, 1}, 2, true, zeros);

  // Constants
  Tensor *two = init(&a, &ac, device, FLOAT32, (u64[]){1}, 1, false, NULL);
  float two_val = 2.0f;
  copy_h2d(two, &two_val);

  // Define graph
  u64 h1_shape[] = {4, hidden_size};
  Tensor *h1_mm = init(&a, &ac, device, FLOAT32, h1_shape, 2, true, NULL);
  Node *n_h1_mm = arena_node_alloc(&a, (Tensor *[]){X, W1}, 2, h1_mm, get_op_impl(MATMUL), 0, false);

  Tensor *h1 = init(&a, &ac, device, FLOAT32, h1_shape, 2, true, NULL);
  Node *n_h1 = arena_node_alloc(&a, (Tensor *[]){h1_mm, b1}, 2, h1, get_op_impl(ADD), 0, false);

  Tensor *h1_abs = init(&a, &ac, device, FLOAT32, h1_shape, 2, true, NULL);
  Node *n_h1_abs = arena_node_alloc(&a, (Tensor *[]){h1}, 1, h1_abs, get_op_impl(ABS), 0, false);

  Tensor *h1_plus_abs = init(&a, &ac, device, FLOAT32, h1_shape, 2, true, NULL);
  Node *n_h1_plus_abs = arena_node_alloc(&a, (Tensor *[]){h1, h1_abs}, 2, h1_plus_abs, get_op_impl(ADD), 0, false);

  Tensor *a1 = init(&a, &ac, device, FLOAT32, h1_shape, 2, true, NULL);
  Node *n_a1 = arena_node_alloc(&a, (Tensor *[]){h1_plus_abs, two}, 2, a1, get_op_impl(DIV), 0, false);

  u64 logits_shape[] = {4, 1};
  Tensor *logits_mm = init(&a, &ac, device, FLOAT32, logits_shape, 2, true, NULL);
  Node *n_logits_mm = arena_node_alloc(&a, (Tensor *[]){a1, W2}, 2, logits_mm, get_op_impl(MATMUL), 0, false);

  Tensor *logits = init(&a, &ac, device, FLOAT32, logits_shape, 2, true, NULL);
  Node *n_logits = arena_node_alloc(&a, (Tensor *[]){logits_mm, b2}, 2, logits, get_op_impl(ADD), 0, false);

  Tensor *diff = init(&a, &ac, device, FLOAT32, logits_shape, 2, true, NULL);
  Node *n_diff = arena_node_alloc(&a, (Tensor *[]){logits, Y}, 2, diff, get_op_impl(SUB), 0, false);

  Tensor *sq_diff = init(&a, &ac, device, FLOAT32, logits_shape, 2, true, NULL);
  Node *n_sq_diff = arena_node_alloc(&a, (Tensor *[]){diff, diff}, 2, sq_diff, get_op_impl(MUL), 0, false);

  Tensor *loss = init(&a, &ac, device, FLOAT32, (u64[]){1}, 1, true, NULL);
  Node *n_loss = arena_node_alloc(&a, (Tensor *[]){sq_diff}, 1, loss, get_op_impl(MEAN), MAX_NDIM + 1, false);

  SGD optimizer = arena_alloc_sgd(0.01f);
  Tensor *params[] = {W1, b1, W2, b2};
  Tensor *intermediates[] = {h1_mm, h1, h1_abs, h1_plus_abs, a1, logits_mm, logits, diff, sq_diff, loss};

  printf("Starting training on CUDA...\n");
  for (int epoch = 0; epoch < 20000; ++epoch) {
    for (int i = 0; i < 4; ++i) zero_grad_cuda(params[i]);
    for (int i = 0; i < 10; ++i) zero_grad_cuda(intermediates[i]);

    zeros(h1_mm, numel(h1_mm));
    zeros(logits_mm, numel(logits_mm));

    forward(n_loss);
    
    if (epoch % 2000 == 0) {
      float loss_val;
      extern void arena_memcpy_d2h_cuda(void *dest, const void *src, u64 size);
      arena_memcpy_d2h_cuda(&loss_val, loss->data, sizeof(float));
      printf("Epoch %d, Loss: %.6f\n", epoch, loss_val);
    }

    set_ones_grad(loss);
    backward(n_loss);

    sgd_step_cuda(&optimizer, params, 4);
  }

  printf("\nFinal Predictions:\n");
  zeros(h1_mm, numel(h1_mm));
  zeros(logits_mm, numel(logits_mm));
  forward(n_loss);
  print_tensor(logits, "Predictions");
  print_tensor(Y, "Targets");
  
  float final_loss;
  extern void arena_memcpy_d2h_cuda(void *dest, const void *src, u64 size);
  arena_memcpy_d2h_cuda(&final_loss, loss->data, sizeof(float));
  printf("Final Loss: %.6f\n", final_loss);

  arena_release(&a);
  arena_release(&ac);

  return 0;
}

