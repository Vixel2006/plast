#include "kernels/flatten.h"
#include "kernels/cpu_utils.h"
#include "core/tensor.h"
#include <string.h>

void flatten_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params) {
  const Tensor *a = inputs[0];
  output->data = a->data;
  output->dtype = a->dtype;
  output->device = a->device;
  output->requires_grad = a->requires_grad;
  output->grad = a->grad;
  // output->shape / output->ndim already set by tensor_init
  compute_view_strides(a->shape, a->strides, a->ndim, output->shape, output->ndim, output->strides);
}

void flatten_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params) {
}
