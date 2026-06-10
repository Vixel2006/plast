#include "core/arena.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "kernels/conv2d.h"
#include "kernels/matmul.h"
#include "kernels/ops/shape.h"
#include "core/op.h"
#include <stdarg.h>
#include <string.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

extern "C" void conv_relu_cuda_forward(const Tensor **inputs, Tensor *output,
                                        KernelParams params) {
  Arena *a = (Arena *)params.dim;
  u64 stride = params.keepdim;
  float alpha = params.fval;

  const Tensor *a_input = inputs[0];
  const Tensor *kernel = inputs[1];

  Tensor *flattened_kernel_view = (Tensor *)arena_alloc(a, sizeof(Tensor), 8);
  memset(flattened_kernel_view, 0, sizeof(Tensor));

  Op flatten_op = get_op_impl(FLATTEN);
  ForwardKernel flatten_kernel = forward_kernel_dispatcher(flatten_op, CUDA);
  const Tensor *flatten_inputs[1] = {kernel};
  flatten_kernel(flatten_inputs, flattened_kernel_view, (KernelParams){0, 0});

  u64 N = a_input->shape[0];
  u64 C = a_input->shape[1];
  u64 H_in = a_input->shape[2];
  u64 W_in = a_input->shape[3];

  u64 kh = kernel->shape[2];
  u64 kw = kernel->shape[3];

  u64 H_out = (H_in - kh) / stride + 1;
  u64 W_out = (W_in - kw) / stride + 1;

  u64 im2col_output_shape[2] = {N * H_out * W_out, C * kh * kw};
  u64 *im2col_output_strides = compute_strides(im2col_output_shape, 2);
  Tensor *im2col_output = arena_tensor_alloc(a, a, im2col_output_shape, 2, im2col_output_strides,
                                             a_input->dtype, a_input->requires_grad, NULL, CUDA);
  free(im2col_output_strides);

  launch_im2col_cuda_float((const float *)a_input->data, (float *)im2col_output->data,
                           N, C, H_in, W_in, kh, kw, stride,
                           a_input->strides[0], a_input->strides[1],
                           a_input->strides[2], a_input->strides[3]);
  cudaDeviceSynchronize();

  Tensor *flattened_kernel_transposed = (Tensor *)arena_alloc(a, sizeof(Tensor), 8);
  memset(flattened_kernel_transposed, 0, sizeof(Tensor));

  Op transpose_op = get_op_impl(TRANSPOSE);
  ForwardKernel transpose_kernel = forward_kernel_dispatcher(transpose_op, CUDA);
  const Tensor *transpose_inputs[1] = {flattened_kernel_view};
  transpose_kernel(transpose_inputs, flattened_kernel_transposed, (KernelParams){0, 1});

  // Use matmul_relu for fused activation
  Op mm_relu_op = get_op_impl(MATMUL_RELU);
  ForwardKernel mm_relu_kernel = forward_kernel_dispatcher(mm_relu_op, CUDA);
  const Tensor *mm_inputs[2] = {im2col_output, flattened_kernel_transposed};
  KernelParams mm_params = {.dim = 0, .keepdim = 0, .fval = alpha};
  mm_relu_kernel(mm_inputs, output, mm_params);
}

extern "C" void conv_relu_cuda_backward(Tensor **inputs, const Tensor *output,
                                         KernelParams params) {
  Arena *a = (Arena *)params.dim;
  u64 stride = params.keepdim;
  float alpha = params.fval;

  Tensor *a_input = inputs[0];
  Tensor *kernel = inputs[1];

  u64 N = a_input->shape[0];
  u64 C = a_input->shape[1];
  u64 H_in = a_input->shape[2];
  u64 W_in = a_input->shape[3];

  u64 kh = kernel->shape[2];
  u64 kw = kernel->shape[3];

  u64 H_out = (H_in - kh) / stride + 1;
  u64 W_out = (W_in - kw) / stride + 1;

  u64 numel_out = numel(output);
  float *dc_mod = NULL;
  if (cudaMalloc(&dc_mod, numel_out * sizeof(float)) != cudaSuccess)
    return;

  int block_size = 256;
  int grid_size = CEIL_DIV(numel_out, (u64)block_size);

  extern void launch_relu_grad_modulate_cuda(const float *, const float *, float *,
                                              u64, float, int, int);
  launch_relu_grad_modulate_cuda((const float *)output->grad->data,
                                  (const float *)output->data,
                                  dc_mod, numel_out, alpha, grid_size, block_size);

  Tensor *flattened_kernel_view = (Tensor *)arena_alloc(a, sizeof(Tensor), 8);
  memset(flattened_kernel_view, 0, sizeof(Tensor));

  Op flatten_op = get_op_impl(FLATTEN);
  ForwardKernel flatten_kernel = forward_kernel_dispatcher(flatten_op, CUDA);
  const Tensor *flatten_inputs[1] = {kernel};
  flatten_kernel(flatten_inputs, flattened_kernel_view, (KernelParams){0, 0});

  // Gradient w.r.t. input
  u64 grad_im2col_output_shape[2] = {N * H_out * W_out, C * kh * kw};
  u64 *grad_im2col_output_strides = compute_strides(grad_im2col_output_shape, 2);
  Tensor *grad_im2col_output =
      arena_tensor_alloc(a, a, grad_im2col_output_shape, 2, grad_im2col_output_strides,
                         output->dtype, false, NULL, CUDA);
  free(grad_im2col_output_strides);

  Tensor dc_mod_tensor;
  memset(&dc_mod_tensor, 0, sizeof(Tensor));
  dc_mod_tensor.data = dc_mod;
  dc_mod_tensor.ndim = output->ndim;
  memcpy(dc_mod_tensor.shape, output->shape, output->ndim * sizeof(u64));
  memcpy(dc_mod_tensor.strides, output->strides, output->ndim * sizeof(u64));
  dc_mod_tensor.dtype = output->dtype;
  dc_mod_tensor.device = CUDA;

  Op matmul_op = get_op_impl(MATMUL);
  ForwardKernel matmul_kernel = forward_kernel_dispatcher(matmul_op, CUDA);
  const Tensor *matmul_inputs_grad_input[2] = {&dc_mod_tensor, flattened_kernel_view};
  matmul_kernel(matmul_inputs_grad_input, grad_im2col_output, (KernelParams){0, 0});

  if (a_input->grad) {
    zeros(a_input->grad, numel(a_input->grad));
  }

  launch_col2im_cuda_float((const float *)grad_im2col_output->data,
                           (float *)a_input->grad->data,
                           N, C, H_in, W_in, kh, kw, stride,
                           a_input->strides[0], a_input->strides[1],
                           a_input->strides[2], a_input->strides[3]);
  cudaDeviceSynchronize();

  // Gradient w.r.t. kernel
  u64 im2col_output_shape[2] = {N * H_out * W_out, C * kh * kw};
  u64 *im2col_output_strides = compute_strides(im2col_output_shape, 2);
  Tensor *im2col_output = arena_tensor_alloc(a, a, im2col_output_shape, 2,
                                             im2col_output_strides, a_input->dtype, false, NULL, CUDA);
  free(im2col_output_strides);

  launch_im2col_cuda_float((const float *)a_input->data, (float *)im2col_output->data,
                           N, C, H_in, W_in, kh, kw, stride,
                           a_input->strides[0], a_input->strides[1],
                           a_input->strides[2], a_input->strides[3]);
  cudaDeviceSynchronize();

  Tensor *output_grad_transposed = (Tensor *)arena_alloc(a, sizeof(Tensor), 8);
  memset(output_grad_transposed, 0, sizeof(Tensor));

  Op transpose_op = get_op_impl(TRANSPOSE);
  ForwardKernel transpose_kernel = forward_kernel_dispatcher(transpose_op, CUDA);
  const Tensor *transpose_inputs_output_grad[1] = {&dc_mod_tensor};
  transpose_kernel(transpose_inputs_output_grad, output_grad_transposed, (KernelParams){0, 1});

  u64 grad_flattened_kernel_shape[2] = {kernel->shape[0],
                                        kernel->shape[1] * kernel->shape[2] * kernel->shape[3]};
  u64 *grad_flattened_kernel_strides = compute_strides(grad_flattened_kernel_shape, 2);
  Tensor *grad_flattened_kernel =
      arena_tensor_alloc(a, a, grad_flattened_kernel_shape, 2, grad_flattened_kernel_strides,
                         output->dtype, false, NULL, CUDA);
  free(grad_flattened_kernel_strides);

  const Tensor *matmul_inputs_grad_kernel[2] = {output_grad_transposed, im2col_output};
  matmul_kernel(matmul_inputs_grad_kernel, grad_flattened_kernel, (KernelParams){0, 0});

  if (kernel->grad) {
    zeros(kernel->grad, numel(kernel->grad));
  }

  cudaMemcpy(kernel->grad->data, grad_flattened_kernel->data,
             numel(kernel->grad) * dtype_size(kernel->grad->dtype),
             cudaMemcpyDeviceToDevice);

  cudaFree(dc_mod);
}
