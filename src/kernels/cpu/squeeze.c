#include "kernels/squeeze.h"
#include "kernels/cpu_utils.h"
#include "core/tensor.h"
#include <string.h>

void squeeze_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params) {
  const Tensor *a = inputs[0];
  u64 axis = params.dim;

  output->data = a->data;
  output->dtype = a->dtype;
  output->device = a->device;
  output->requires_grad = a->requires_grad;
  output->grad = a->grad;

  u64 new_ndim;
  compute_squeeze_shape_strides(a->shape, a->strides, a->ndim, axis, output->shape, output->strides,
                                &new_ndim);
  output->ndim = new_ndim;
}

void squeeze_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params) {
}
