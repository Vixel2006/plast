#include "kernels/conv2d.h"
#include "core/arena.h" // For arena_alloc
#include "kernels/ops/shape.h"
#include "kernels/matmul.h"
#include "core/op.h"
#include <stdarg.h>
#include <string.h> // For memset

void im2col_cpu_float_kernel(float *img, float *buffer, u64 *kernel_size, const u64 *img_shape,
                             const u64 *img_strides, u64 img_ndim, u64 stride) {
  u64 kh = kernel_size[0];
  u64 kw = kernel_size[1];
  u64 N = img_shape[0];
  u64 C = img_shape[1];
  u64 H_in = img_shape[2];
  u64 W_in = img_shape[3];

  u64 H_out = (H_in - kh) / stride + 1;
  u64 W_out = (W_in - kw) / stride + 1;

#pragma omp parallel for collapse(4)
  for (u64 batch = 0; batch < N; ++batch) {
    for (u64 out_h = 0; out_h < H_out; ++out_h) {
      for (u64 out_w = 0; out_w < W_out; ++out_w) {
        for (u64 c = 0; c < C; ++c) {
          u64 start_h = out_h * stride;
          u64 start_w = out_w * stride;

          u64 buffer_base_idx = batch * (H_out * W_out * C * kh * kw) +
                                out_h * (W_out * C * kh * kw) + out_w * (C * kh * kw) +
                                c * (kh * kw);

          float *img_channel_base = img + batch * img_strides[0] + c * img_strides[1];

          for (u64 kr = 0; kr < kh; ++kr) {
            for (u64 kc = 0; kc < kw; ++kc) {
              u64 img_offset_in_channel =
                  (start_h + kr) * img_strides[2] + (start_w + kc) * img_strides[3];

              buffer[buffer_base_idx + kr * kw + kc] = img_channel_base[img_offset_in_channel];
            }
          }
        }
      }
    }
  }
}

void col2im_cpu_float_kernel(float *buffer, float *img, u64 *kernel_size, const u64 *img_shape,
                             const u64 *img_strides, u64 img_ndim, u64 stride) {
  u64 kh = kernel_size[0];
  u64 kw = kernel_size[1];
  u64 N = img_shape[0];
  u64 C = img_shape[1];
  u64 H_in = img_shape[2]; // This is the original image height
  u64 W_in = img_shape[3]; // This is the original image width

  u64 H_out = (H_in - kh) / stride + 1;
  u64 W_out = (W_in - kw) / stride + 1;

  // Ensure img is zero-initialized before this kernel is called for correct
  // accumulation.

#pragma omp parallel for collapse(4)
  for (u64 batch = 0; batch < N; ++batch) {
    for (u64 out_h = 0; out_h < H_out; ++out_h) {
      for (u64 out_w = 0; out_w < W_out; ++out_w) {
        for (u64 c = 0; c < C; ++c) {
          u64 start_h = out_h * stride;
          u64 start_w = out_w * stride;

          u64 buffer_base_idx = batch * (H_out * W_out * C * kh * kw) +
                                out_h * (W_out * C * kh * kw) + out_w * (C * kh * kw) +
                                c * (kh * kw);

          float *img_channel_base = img + batch * img_strides[0] + c * img_strides[1];

          for (u64 kr = 0; kr < kh; ++kr) {
            for (u64 kc = 0; kc < kw; ++kc) {
              u64 img_offset_in_channel =
                  (start_h + kr) * img_strides[2] + (start_w + kc) * img_strides[3];

#pragma omp atomic
              img_channel_base[img_offset_in_channel] += buffer[buffer_base_idx + kr * kw + kc];
            }
          }
        }
      }
    }
  }
}

void conv2d_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params) {
  Arena *a = (Arena *)params.dim;
  u64 stride = params.keepdim;

  const Tensor *a_input = inputs[0]; // Input image
  const Tensor *kernel = inputs[1];  // Convolution kernel

  // 1. Flatten the kernel (keep batch dim, flatten spatial/channel dims)
  Tensor *flattened_kernel_view = (Tensor *)arena_alloc(a, sizeof(Tensor), 8);
  memset(flattened_kernel_view, 0, sizeof(Tensor));
  flattened_kernel_view->ndim = 2;
  flattened_kernel_view->shape[0] = kernel->shape[0];
  flattened_kernel_view->shape[1] = kernel->shape[1] * kernel->shape[2] * kernel->shape[3];

  Op flatten_op = get_op_impl(FLATTEN);
  ForwardKernel flatten_kernel = forward_kernel_dispatcher(flatten_op, CPU);
  const Tensor *flatten_inputs[1] = {kernel};
  flatten_kernel(flatten_inputs, flattened_kernel_view, (KernelParams){0, 0});

  // 2. Perform im2col on the input 'a_input'
  u64 N = a_input->shape[0];
  u64 C = a_input->shape[1];
  u64 H_in = a_input->shape[2];
  u64 W_in = a_input->shape[3];

  u64 kh = kernel->shape[2]; // Kernel height
  u64 kw = kernel->shape[3]; // Kernel width

  u64 H_out = (H_in - kh) / stride + 1;
  u64 W_out = (W_in - kw) / stride + 1;

  u64 im2col_output_shape[2] = {N * H_out * W_out, C * kh * kw};
  u64 *im2col_output_strides = compute_strides(im2col_output_shape, 2);
  Tensor *im2col_output = arena_tensor_alloc(a, a, im2col_output_shape, 2, im2col_output_strides,
                                             a_input->dtype, a_input->requires_grad, NULL, CPU);
  free(im2col_output_strides);

  u64 kernel_size_arr[2] = {kh, kw};
  im2col_cpu_float_kernel((float *)a_input->data, (float *)im2col_output->data, kernel_size_arr,
                          a_input->shape, a_input->strides, a_input->ndim, stride);

  // 3. Transpose the flattened kernel
  Tensor *flattened_kernel_transposed = (Tensor *)arena_alloc(a, sizeof(Tensor), 8);
  memset(flattened_kernel_transposed, 0, sizeof(Tensor));

  Op transpose_op = get_op_impl(TRANSPOSE);
  ForwardKernel transpose_kernel = forward_kernel_dispatcher(transpose_op, CPU);
  const Tensor *transpose_inputs[1] = {flattened_kernel_view};
  transpose_kernel(transpose_inputs, flattened_kernel_transposed, (KernelParams){0, 1});

  // 4. Perform matmul: output = im2col_output @ flattened_kernel_transposed
  Op matmul_op = get_op_impl(MATMUL);
  ForwardKernel matmul_kernel = forward_kernel_dispatcher(matmul_op, CPU);
  const Tensor *matmul_inputs[2] = {im2col_output, flattened_kernel_transposed};
  matmul_kernel(matmul_inputs, output, (KernelParams){0, 0});
}

void conv2d_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params) {
  Arena *a = (Arena *)params.dim;
  u64 stride = params.keepdim;

  Tensor *a_input = inputs[0]; // Input image
  Tensor *kernel = inputs[1];  // Convolution kernel

  // output->grad contains the gradient from the subsequent layer.

  u64 N = a_input->shape[0];
  u64 C = a_input->shape[1];
  u64 H_in = a_input->shape[2];
  u64 W_in = a_input->shape[3];

  u64 kh = kernel->shape[2]; // Kernel height
  u64 kw = kernel->shape[3]; // Kernel width

  u64 H_out = (H_in - kh) / stride + 1;
  u64 W_out = (W_in - kw) / stride + 1;

  // 1. Gradient with respect to the input (a_input->grad)
  // Create a temporary tensor to hold the flattened view of the kernel
  Tensor *flattened_kernel_view = (Tensor *)arena_alloc(a, sizeof(Tensor), 8);
  memset(flattened_kernel_view, 0, sizeof(Tensor));
  flattened_kernel_view->ndim = 2;
  flattened_kernel_view->shape[0] = kernel->shape[0];
  flattened_kernel_view->shape[1] = kernel->shape[1] * kernel->shape[2] * kernel->shape[3];

  Op flatten_op = get_op_impl(FLATTEN);
  ForwardKernel flatten_kernel = forward_kernel_dispatcher(flatten_op, CPU);
  const Tensor *flatten_inputs[1] = {kernel};
  flatten_kernel(flatten_inputs, flattened_kernel_view, (KernelParams){0, 0});

  u64 grad_im2col_output_shape[2] = {N * H_out * W_out, C * kh * kw};
  u64 *grad_im2col_output_strides = compute_strides(grad_im2col_output_shape, 2);
  Tensor *grad_im2col_output = arena_tensor_alloc(
      a, a, grad_im2col_output_shape, 2, grad_im2col_output_strides, output->dtype, false, NULL,
      CPU); // grad_im2col_output does not require grad
  free(grad_im2col_output_strides);

  // output->grad is [N * H_out * W_out, F], flattened_kernel_view is [F, C * kh * kw]
  // Performing matmul: grad_im2col_output = output->grad @ flattened_kernel_view
  Op matmul_op = get_op_impl(MATMUL);
  ForwardKernel matmul_kernel = forward_kernel_dispatcher(matmul_op, CPU);
  const Tensor *matmul_inputs_grad_input[2] = {output->grad, flattened_kernel_view};
  matmul_kernel(matmul_inputs_grad_input, grad_im2col_output, (KernelParams){0, 0});

  // Ensure a_input->grad is zero-initialized before accumulation
  if (a_input->grad) {
    memset(a_input->grad->data, 0, numel(a_input->grad) * dtype_size(a_input->grad->dtype));
  }

  u64 kernel_size_arr[2] = {kh, kw};
  col2im_cpu_float_kernel((float *)grad_im2col_output->data, (float *)a_input->grad->data,
                          kernel_size_arr, a_input->shape, a_input->strides, a_input->ndim, stride);

  // 2. Gradient with respect to the kernel (kernel->grad)
  u64 im2col_output_shape[2] = {N * H_out * W_out, C * kh * kw};
  u64 *im2col_output_strides = compute_strides(im2col_output_shape, 2);
  Tensor *im2col_output =
      arena_tensor_alloc(a, a, im2col_output_shape, 2, im2col_output_strides, a_input->dtype, false,
                         NULL, CPU); // im2col_output does not require grad
  free(im2col_output_strides);

  im2col_cpu_float_kernel((float *)a_input->data, (float *)im2col_output->data, kernel_size_arr,
                          a_input->shape, a_input->strides, a_input->ndim, stride);

  Tensor *output_grad_transposed = (Tensor *)arena_alloc(a, sizeof(Tensor), 8);
  memset(output_grad_transposed, 0, sizeof(Tensor));

  Op transpose_op = get_op_impl(TRANSPOSE);
  ForwardKernel transpose_kernel = forward_kernel_dispatcher(transpose_op, CPU);
  const Tensor *transpose_inputs_output_grad[1] = {output->grad};
  transpose_kernel(transpose_inputs_output_grad, output_grad_transposed, (KernelParams){0, 1});

  u64 grad_flattened_kernel_shape[2] = {kernel->shape[0],
                                        kernel->shape[1] * kernel->shape[2] * kernel->shape[3]};
  u64 *grad_flattened_kernel_strides = compute_strides(grad_flattened_kernel_shape, 2);
  Tensor *grad_flattened_kernel =
      arena_tensor_alloc(a, a, grad_flattened_kernel_shape, 2, grad_flattened_kernel_strides,
                         output->dtype, false, NULL,
                         CPU); // grad_flattened_kernel does not require grad
  free(grad_flattened_kernel_strides);

  const Tensor *matmul_inputs_grad_kernel[2] = {output_grad_transposed, im2col_output};
  matmul_kernel(matmul_inputs_grad_kernel, grad_flattened_kernel, (KernelParams){0, 0});

  // Ensure kernel->grad is zero-initialized before accumulation
  if (kernel->grad) {
    memset(kernel->grad->data, 0, numel(kernel->grad) * dtype_size(kernel->grad->dtype));
  }

  // Copy data from grad_flattened_kernel to kernel->grad
  memcpy(kernel->grad->data, grad_flattened_kernel->data,
         numel(kernel->grad) * dtype_size(kernel->grad->dtype));
}
