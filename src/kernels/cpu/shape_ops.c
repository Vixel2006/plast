#include "kernels/ops/shape.h"
#include "kernels/cpu_utils.h"
#include "kernels/kernel_macros.h"
#include <string.h>

DEFINE_SHAPE_OP(view, compute_view_strides, a->shape, a->strides, a->ndim, output->shape,
                output->ndim, output->strides)
DEFINE_SHAPE_OP(flatten, compute_view_strides, a->shape, a->strides, a->ndim, output->shape,
                output->ndim, output->strides)
DEFINE_SHAPE_OP(squeeze, compute_squeeze_shape_strides, a->shape, a->strides, a->ndim, params.dim,
                output->shape, output->strides, &output->ndim)
DEFINE_SHAPE_OP(unsqueeze, compute_unsqueeze_shape_strides, a->shape, a->strides, a->ndim,
                params.dim, output->shape, output->strides, &output->ndim)
DEFINE_SHAPE_OP(expand, compute_expand_strides, a->shape, a->strides, a->ndim, output->shape,
                output->ndim, output->strides)
DEFINE_SHAPE_OP(broadcast, compute_broadcast_strides, a->shape, a->strides, a->ndim, output->shape,
                output->ndim, output->strides)

void transpose_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params) {
  const Tensor *a = inputs[0];
  u64 axis1 = params.dim;
  u64 axis2 = params.keepdim;

  output->data = a->data;
  output->dtype = a->dtype;
  output->device = a->device;
  output->requires_grad = a->requires_grad;
  output->grad = a->grad;

  output->ndim = a->ndim;
  memcpy(output->shape, a->shape, a->ndim * sizeof(u64));
  memcpy(output->strides, a->strides, a->ndim * sizeof(u64));

  u64 temp = output->shape[axis1];
  output->shape[axis1] = output->shape[axis2];
  output->shape[axis2] = temp;

  temp = output->strides[axis1];
  output->strides[axis1] = output->strides[axis2];
  output->strides[axis2] = temp;
}

void transpose_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params) {
}
