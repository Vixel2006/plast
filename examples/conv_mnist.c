// Conv2D + Dense MNIST example
// Build: gcc -O3 -I../include conv_mnist.c -L.. -lplast -lm -lgomp -o conv_mnist
#include "plast/plast.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Dummy MNIST data for demonstration.
// In a real use case, load actual MNIST images.
static void gen_dummy_data(float *images, float *labels, int n) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < 784; ++j)
      images[i * 784 + j] = ((float)rand() / RAND_MAX) * 0.5f + 0.1f;
    int label = rand() % 10;
    for (int j = 0; j < 10; ++j)
      labels[i * 10 + j] = (j == label) ? 1.0f : 0.0f;
  }
}

int main() {
  srand(42);

  // Model: Conv2D(1→8, 3x3) → ReLU → Conv2D(8→16, 3x3) → ReLU → Flatten → Dense(16*24*24→10) → Softmax
  // Input: [N, 1, 28, 28]
  // After first conv (stride=1, kernel=3): [N, 8, 26, 26]
  // After second conv (stride=1, kernel=3): [N, 16, 24, 24]
  // After flatten: [N, 16*24*24] = [N, 9216]
  // After dense: [N, 10]
  // After softmax: [N, 10]

  PlastModel *m = plast_model_create(CPU);
  plast_model_add_conv2d(m, 8, 3, 1, true);     // out_channels=8, kernel=3, stride=1
  plast_model_add_activation(m, PLAST_ACT_RELU);
  plast_model_add_conv2d(m, 16, 3, 1, true);    // out_channels=16, kernel=3, stride=1
  plast_model_add_activation(m, PLAST_ACT_RELU);
  plast_model_add_flatten(m);
  plast_model_add_dense(m, 10, true);
  plast_model_add_activation(m, PLAST_ACT_SOFTMAX);

  u64 input_shape[] = {4, 1, 28, 28}; // batch=4, channels=1, H=28, W=28
  plast_model_compile(m, input_shape, 4);
  plast_model_summary(m);

  // Generate dummy data
  int n = 4;
  float *images = (float *)malloc(n * 784 * sizeof(float));
  float *labels = (float *)malloc(n * 10 * sizeof(float));
  gen_dummy_data(images, labels, n);

  // Create optimizer
  PlastOptimizer *opt = plast_optim_sgd_create(0.01f);
  plast_model_set_optimizer(m, opt);

  // Create target tensor once
  Tensor *target = plast_tensor_from_array(labels, (u64[]){(u64)n, 10}, 2);

  printf("\nDummy training (ConvNet)...\n");
  for (int epoch = 0; epoch < 5; ++epoch) {
    Tensor *in = plast_model_input(m);
    memcpy(in->data, images, n * 784 * sizeof(float));

    Tensor *pred = plast_model_forward(m);

    PlastLoss loss = plast_cross_entropy_loss(pred, target, plast_model_meta_arena(m),
                                              plast_model_data_arena(m));

    plast_model_zero_grad(m);
    plast_model_backward(m, loss.output);
    plast_optimizer_step(m);

    float *loss_data = (float *)loss.output->data;
    printf("  Epoch %d, Loss: %.4f\n", epoch, loss_data[0]);
  }

  free(target->data);
  free(target);

  float *pred_data = (float *)plast_model_output(m)->data;
  printf("\nSample predictions:\n");
  for (int i = 0; i < n; ++i) {
    int pred_class = 0;
    for (int j = 1; j < 10; ++j) {
      if (pred_data[i * 10 + j] > pred_data[i * 10 + pred_class])
        pred_class = j;
    }
    int true_class = 0;
    for (int j = 1; j < 10; ++j) {
      if (labels[i * 10 + j] > labels[i * 10 + true_class])
        true_class = j;
    }
    printf("  Sample %d: predicted=%d, true=%d\n", i, pred_class, true_class);
  }

  // Save model weights
  plast_model_save(m, "conv_mnist_weights.plast");
  printf("\nWeights saved to conv_mnist_weights.plast\n");

  free(images);
  free(labels);
  plast_model_free(m);
  return 0;
}
