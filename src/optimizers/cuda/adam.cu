#include "adam.h"
#include "cuda_utils.cuh"

__global__ void adam_kernel(float *data, const float *grad, float *m_data, float *v_data,
                            float learning_rate, float beta1, float beta2, float epsilon, int t,
                            size_t num_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float lr_t = learning_rate * rsqrtf(1.0f - powf(beta2, t)) / (1.0f - powf(beta1, t));

    m_data[idx] = beta1 * m_data[idx] + (1.0f - beta1) * grad[idx];
    v_data[idx] = beta2 * v_data[idx] + (1.0f - beta2) * grad[idx] * grad[idx];
    data[idx] -= lr_t * m_data[idx] / (sqrtf(v_data[idx]) + epsilon);
  }
}

void adam_step_cuda(Adam *optimizer, Tensor **parameters, int num_parameters) {
}
