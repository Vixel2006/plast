// RL Policy Network example — continuous + discrete action policies
// Build: gcc -O3 -I../include rl_policy_network.c -L.. -lplast -lm -lgomp -o rl_policy
#include "plast/plast.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Simple continuous policy: state → Tanh → scaled actions
void continuous_policy_example() {
  printf("=== Continuous Policy Network (e.g. MuJoCo) ===\n");
  printf("  state_dim=8, action_dim=2, hidden=64\n\n");

  PlastModel *m = plast_model_create(CPU);
  plast_model_add_dense(m, 64, true);
  plast_model_add_activation(m, PLAST_ACT_TANH);
  plast_model_add_dense(m, 64, true);
  plast_model_add_activation(m, PLAST_ACT_TANH);
  plast_model_add_dense(m, 2, true);
  plast_model_add_activation(m, PLAST_ACT_TANH); // bounded actions [-1, 1]

  u64 input_shape[] = {1, 8}; // batch=1, state_dim=8
  plast_model_compile(m, input_shape, 2);
  plast_model_summary(m);

  // Random state input
  float state[] = {0.1, -0.3, 0.5, -0.7, 0.2, -0.1, 0.9, -0.4};

  Tensor *in = plast_model_input(m);
  memcpy(in->data, state, 8 * sizeof(float));

  Tensor *action = plast_model_forward(m);
  plast_tensor_print(action, "Continuous Action");

  // Training step (PPO-style: forward, compute loss, backward, step)
  float reward = 1.0f; // dummy reward for demonstration
  printf("  Simulated reward: %.2f\n\n", reward);

  plast_model_free(m);
}

// Discrete policy: state → Dense → ReLU → Softmax → action probabilities
void discrete_policy_example() {
  printf("=== Discrete Policy Network (e.g. CartPole, Atari) ===\n");
  printf("  state_dim=4, num_actions=3, hidden=32\n\n");

  PlastModel *m = plast_model_create(CPU);
  plast_model_add_dense(m, 32, true);
  plast_model_add_activation(m, PLAST_ACT_RELU);
  plast_model_add_dense(m, 32, true);
  plast_model_add_activation(m, PLAST_ACT_RELU);
  plast_model_add_dense(m, 3, true);          // 3 discrete actions (logits)
  plast_model_add_activation(m, PLAST_ACT_SOFTMAX); // action probabilities

  u64 input_shape[] = {1, 4};
  plast_model_compile(m, input_shape, 2);
  plast_model_summary(m);

  float state[] = {0.1, -0.2, 0.3, -0.4};

  Tensor *in = plast_model_input(m);
  memcpy(in->data, state, 4 * sizeof(float));

  Tensor *probs = plast_model_forward(m);
  plast_tensor_print(probs, "Action Probabilities");

  // Sample action
  float *p = (float *)probs->data;
  float r = (float)rand() / RAND_MAX;
  float cum = 0.0f;
  int chosen = 0;
  for (int i = 0; i < 3; ++i) {
    cum += p[i];
    if (r < cum) {
      chosen = i;
      break;
    }
  }
  printf("  Sampled action: %d (probs: [%.3f, %.3f, %.3f])\n\n", chosen, p[0], p[1], p[2]);

  plast_model_free(m);
}

int main() {
  srand(42);
  continuous_policy_example();
  discrete_policy_example();
  return 0;
}
