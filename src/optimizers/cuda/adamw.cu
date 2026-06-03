#include "adamw.h"
#include "cuda_utils.cuh"

__global__ void adamw_kernel(float *data, const float *grad, float *m_data, float *v_data,
                             float learning_rate, float beta1, float beta2, float epsilon,
                             float weight_decay, int t, size_t num_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    // Apply weight decay
    data[idx] -= learning_rate * weight_decay * data[idx];

    float lr_t = learning_rate * rsqrtf(1.0f - powf(beta2, t)) / (1.0f - powf(beta1, t));

    m_data[idx] = beta1 * m_data[idx] + (1.0f - beta1) * grad[idx];
    v_data[idx] = beta2 * v_data[idx] + (1.0f - beta2) * grad[idx] * grad[idx];
    data[idx] -= lr_t * m_data[idx] / (sqrtf(v_data[idx]) + epsilon);
  }
}

void adamw_step_cuda(AdamW *optimizer, Tensor **parameters, int num_parameters) {
}
