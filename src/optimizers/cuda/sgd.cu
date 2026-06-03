#include "cuda_utils.cuh"
#include "sgd.h"

__global__ void sgd_kernel(float *data, const float *grad, float lr, size_t num_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    data[idx] -= lr * grad[idx];
  }
}

void sgd_step_cuda(SGD *optimizer, Tensor **parameters, int num_parameters) {
  if (optimizer == NULL) {
    fprintf(stderr, "SGD is NULL\n");
    return;
  }

  for (int i = 0; i < num_parameters; ++i) {
    Tensor *param = parameters[i];
    if (param == NULL || param->grad == NULL) {
      continue;
    }

    float *data_d = (float *)param->data;
    float *grad_d = (float *)param->grad->data;
    size_t num_elements = numel(param);

    int blockSize = 256;
    int numBlocks = (num_elements + blockSize - 1) / blockSize;

    sgd_kernel<<<numBlocks, blockSize>>>(data_d, grad_d, optimizer->lr, num_elements);
  }
}
