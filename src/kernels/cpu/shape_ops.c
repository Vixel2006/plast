#include "kernels/ops/shape.h"
#include "kernels/cpu_utils.h"
#include "kernels/kernel_macros.h"
#include <string.h>
#include <stdlib.h>

#ifdef CUDA_AVAILABLE
void reshape_backward_cuda(float *da, const float *dout, u64 num_elements);
void transpose_backward_cuda(float *da, const float *dout, const u64 *a_shape,
                             const u64 *da_strides, const u64 *dout_strides, u64 ndim, u64 axis1,
                             u64 axis2, u64 num_elements);
#endif

static inline void reshape_backward_dispatch(Tensor *a, const Tensor *output, u64 num_elements) {
  if (a->device == CUDA) {
#ifdef CUDA_AVAILABLE
    reshape_backward_cuda((float *)a->grad->data, (const float *)output->grad->data, num_elements);
#endif
  } else {
    float *da = (float *)a->grad->data;
    const float *dout = (const float *)output->grad->data;
    for (u64 i = 0; i < num_elements; ++i) {
      da[i] += dout[i];
    }
  }
}

#define DEFINE_RESHAPE_OP(op_name, compute_fn, ...)                                                \
  void op_name##_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params) {         \
    const Tensor *a = inputs[0];                                                                   \
    output->data = a->data;                                                                        \
    output->dtype = a->dtype;                                                                      \
    output->device = a->device;                                                                    \
    output->requires_grad = a->requires_grad;                                                      \
    compute_fn(__VA_ARGS__);                                                                       \
  }                                                                                                \
  void op_name##_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params) {        \
    Tensor *a = inputs[0];                                                                         \
    if (!a->requires_grad || !output->grad)                                                        \
      return;                                                                                      \
    u64 num_elements = numel(a);                                                                   \
    reshape_backward_dispatch(a, output, num_elements);                                            \
  }

DEFINE_RESHAPE_OP(view, compute_view_strides, a->shape, a->strides, a->ndim, output->shape,
                  output->ndim, output->strides)
DEFINE_RESHAPE_OP(flatten, compute_view_strides, a->shape, a->strides, a->ndim, output->shape,
                  output->ndim, output->strides)
DEFINE_RESHAPE_OP(squeeze, compute_view_strides, a->shape, a->strides, a->ndim, output->shape,
                  output->ndim, output->strides)
DEFINE_RESHAPE_OP(unsqueeze, compute_view_strides, a->shape, a->strides, a->ndim, output->shape,
                  output->ndim, output->strides)

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
  Tensor *a = inputs[0];
  if (!a->requires_grad || !output->grad)
    return;

  u64 axis1 = params.dim;
  u64 axis2 = params.keepdim;

  u64 num_elements = numel(a);
  if (a->device == CUDA) {
#ifdef CUDA_AVAILABLE
    transpose_backward_cuda((float *)a->grad->data, (const float *)output->grad->data, a->shape,
                            a->grad->strides, output->grad->strides, a->ndim, axis1, axis2,
                            num_elements);
#endif
  } else {
    float *da = (float *)a->grad->data;
    const float *dout = (const float *)output->grad->data;

    u64 *coords = (u64 *)malloc(a->ndim * sizeof(u64));
    u64 *coords_transposed = (u64 *)malloc(a->ndim * sizeof(u64));

    for (u64 i = 0; i < num_elements; ++i) {
      linear_to_coords(i, a->shape, a->ndim, coords);
      u64 a_offset = get_offset(coords, a->grad->strides, a->ndim);

      memcpy(coords_transposed, coords, a->ndim * sizeof(u64));
      u64 temp = coords_transposed[axis1];
      coords_transposed[axis1] = coords_transposed[axis2];
      coords_transposed[axis2] = temp;

      u64 out_offset = get_offset(coords_transposed, output->grad->strides, output->ndim);
      da[a_offset] += dout[out_offset];
    }

    free(coords);
    free(coords_transposed);
  }
}
