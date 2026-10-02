// XOR training example — "Hello World" of neural networks
// Build: gcc -O3 -I../include xor.c -L.. -lplast -lm -lgomp -o xor
#include "plast/plast.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main() {
  srand(42);

  // Create model
  PlastModel *m = plast_model_create(CPU);
  plast_model_add_dense(m, 16, true);
  plast_model_add_activation(m, PLAST_ACT_RELU);
  plast_model_add_dense(m, 16, true);
  plast_model_add_activation(m, PLAST_ACT_RELU);
  plast_model_add_dense(m, 1, true);

  // Compile with input shape [4, 2] (batch=4, features=2)
  u64 input_shape[] = {4, 2};
  plast_model_compile(m, input_shape, 2);
  plast_model_summary(m);

  // XOR data
  float x_data[] = {0, 0, 0, 1, 1, 0, 1, 1};
  float y_data[] = {0, 1, 1, 0};

  // Create optimizer
  PlastOptimizer *opt = plast_optim_sgd_create(0.5f);
  plast_model_set_optimizer(m, opt);

  // Create target tensor once
  Tensor *target = plast_tensor_from_array(y_data, (u64[]){4, 1}, 2);

  // Training loop
  printf("\nTraining XOR...\n");
  for (int epoch = 0; epoch <= 5000; ++epoch) {
    Tensor *in = plast_model_input(m);
    memcpy(in->data, x_data, 8 * sizeof(float));

    Tensor *pred = plast_model_forward(m);

    PlastLoss loss = plast_mse_loss(pred, target, plast_model_meta_arena(m),
                                    plast_model_data_arena(m));

    plast_model_zero_grad(m);
    plast_model_backward(m, loss.output);
    plast_optimizer_step(m);

    if (epoch % 1000 == 0) {
      float *loss_data = (float *)loss.output->data;
      printf("Epoch %d, Loss: %.6f\n", epoch, loss_data[0]);
    }
  }

  free(target->data);
  free(target);

  // Final predictions
  printf("\nFinal predictions:\n");
  Tensor *in = plast_model_input(m);
  memcpy(in->data, x_data, 8 * sizeof(float));
  Tensor *pred = plast_model_forward(m);
  float *pred_data = (float *)pred->data;
  for (int i = 0; i < 4; ++i) {
    printf("  XOR(%d, %d) = %.4f (expected %d)\n", (int)x_data[i*2], (int)x_data[i*2+1],
           pred_data[i], (int)y_data[i]);
  }

  // Save weights
  plast_model_save(m, "xor_weights.plast");
  printf("\nWeights saved to xor_weights.plast\n");

  plast_model_free(m);
  return 0;
}
