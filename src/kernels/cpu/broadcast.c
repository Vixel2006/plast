#include "kernels/broadcast.h"
#include "kernels/cpu_utils.h"
#include "tensor.h"
#include <string.h>

void broadcast_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params) {
  const Tensor *a = inputs[0];
  // output->shape / output->ndim already set by tensor_init
  output->data = a->data;
  output->dtype = a->dtype;
  output->device = a->device;
  output->requires_grad = a->requires_grad;
  output->grad = a->grad;
  compute_broadcast_strides(a->shape, a->strides, a->ndim, output->shape, output->ndim,
                            output->strides);
}

void broadcast_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params) {
}
