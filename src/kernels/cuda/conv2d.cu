#include "core/arena.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "kernels/conv2d.h"
#include "kernels/ops/shape.h"
#include "kernels/matmul.h"
#include "core/op.h"
#include <stdarg.h>
#include <string.h>

__global__ void im2col_cuda_float_kernel(const float *img, float *buffer, u64 N, u64 C, u64 H_in,
                                         u64 W_in, u64 kh, u64 kw, u64 stride, u64 img_stride_N,
                                         u64 img_stride_C, u64 img_stride_H, u64 img_stride_W) {
  u64 H_out = (H_in - kh) / stride + 1;
  u64 W_out = (W_in - kw) / stride + 1;

  u64 output_elements = N * H_out * W_out * C * kh * kw;

  u64 idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= output_elements) {
    return;
  }

  u64 col_in_patch = idx % (kh * kw);
  u64 kr = col_in_patch / kw;
  u64 kc = col_in_patch % kw;

  u64 idx_after_kernel = idx / (kh * kw);
  u64 c = idx_after_kernel % C;

  u64 idx_after_channel = idx_after_kernel / C;
  u64 col_in_output_spatial = idx_after_channel % (H_out * W_out);
  u64 out_h = col_in_output_spatial / W_out;
  u64 out_w = col_in_output_spatial % W_out;

  u64 batch = idx_after_channel / (H_out * W_out);

  u64 in_h = out_h * stride + kr;
  u64 in_w = out_w * stride + kc;

  u64 img_idx = batch * img_stride_N + c * img_stride_C + in_h * img_stride_H + in_w * img_stride_W;

  buffer[idx] = img[img_idx];
}

__global__ void col2im_cuda_float_kernel(const float *buffer, float *img, u64 N, u64 C, u64 H_in,
                                         u64 W_in, u64 kh, u64 kw, u64 stride, u64 img_stride_N,
                                         u64 img_stride_C, u64 img_stride_H, u64 img_stride_W) {
  u64 H_out = (H_in - kh) / stride + 1;
  u64 W_out = (W_in - kw) / stride + 1;

  u64 input_buffer_elements = N * H_out * W_out * C * kh * kw;

  u64 idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= input_buffer_elements) {
    return;
  }

  u64 col_in_patch = idx % (kh * kw);
  u64 kr = col_in_patch / kw;
  u64 kc = col_in_patch % kw;

  u64 idx_after_kernel = idx / (kh * kw);
  u64 c = idx_after_kernel % C;

  u64 idx_after_channel = idx_after_kernel / C;
  u64 col_in_output_spatial = idx_after_channel % (H_out * W_out);
  u64 out_h = col_in_output_spatial / W_out;
  u64 out_w = col_in_output_spatial % W_out;

  u64 batch = idx_after_channel / (H_out * W_out);

  u64 in_h = out_h * stride + kr;
  u64 in_w = out_w * stride + kc;

  if (in_h < H_in && in_w < W_in) {
    u64 img_idx =
        batch * img_stride_N + c * img_stride_C + in_h * img_stride_H + in_w * img_stride_W;

    atomicAdd(&img[img_idx], buffer[idx]);
  }
}

void conv2d_cuda_forward(const Tensor **inputs, Tensor *output, KernelParams params) {
  Arena *a = (Arena *)params.dim;
  u64 stride = params.keepdim;

  const Tensor *a_input = inputs[0]; // Input image
  const Tensor *kernel = inputs[1];  // Convolution kernel

  // 1. Flatten the kernel
  Tensor *flattened_kernel_view = (Tensor *)arena_alloc(a, sizeof(Tensor), 8);
  memset(flattened_kernel_view, 0, sizeof(Tensor));

  Op flatten_op = get_op_impl(FLATTEN);
  ForwardKernel flatten_kernel = forward_kernel_dispatcher(flatten_op, CUDA);
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
                                             a_input->dtype, a_input->requires_grad, NULL, CUDA);
  free(im2col_output_strides);

  u64 output_elements = N * H_out * W_out * C * kh * kw;
  int block_size = 256;
  int grid_size = (output_elements + block_size - 1) / block_size;

  im2col_cuda_float_kernel<<<grid_size, block_size>>>(
      (const float *)a_input->data, (float *)im2col_output->data, N, C, H_in, W_in, kh, kw, stride,
      a_input->strides[0], a_input->strides[1], a_input->strides[2], a_input->strides[3]);
  cudaDeviceSynchronize();

  // 3. Transpose the im2col output
  Tensor *im2col_output_transposed = (Tensor *)arena_alloc(a, sizeof(Tensor), 8);
  memset(im2col_output_transposed, 0, sizeof(Tensor));

  Op transpose_op = get_op_impl(TRANSPOSE);
  ForwardKernel transpose_kernel = forward_kernel_dispatcher(transpose_op, CUDA);
  const Tensor *transpose_inputs[1] = {im2col_output};
  transpose_kernel(transpose_inputs, im2col_output_transposed, (KernelParams){0, 1});

  // 4. Perform matmul
  Op matmul_op = get_op_impl(MATMUL);
  ForwardKernel matmul_kernel = forward_kernel_dispatcher(matmul_op, CUDA);
  const Tensor *matmul_inputs[2] = {flattened_kernel_view, im2col_output_transposed};
  matmul_kernel(matmul_inputs, output, (KernelParams){0, 0});
}

void conv2d_cuda_backward(Tensor **inputs, const Tensor *output, KernelParams params) {
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
  Tensor *flattened_kernel_view = (Tensor *)arena_alloc(a, sizeof(Tensor), 8);
  memset(flattened_kernel_view, 0, sizeof(Tensor));

  Op flatten_op = get_op_impl(FLATTEN);
  ForwardKernel flatten_kernel = forward_kernel_dispatcher(flatten_op, CUDA);
  const Tensor *flatten_inputs[1] = {kernel};
  flatten_kernel(flatten_inputs, flattened_kernel_view, (KernelParams){0, 0});

  Tensor *flattened_kernel_transposed = (Tensor *)arena_alloc(a, sizeof(Tensor), 8);
  memset(flattened_kernel_transposed, 0, sizeof(Tensor));

  Op transpose_op = get_op_impl(TRANSPOSE);
  ForwardKernel transpose_kernel = forward_kernel_dispatcher(transpose_op, CUDA);
  const Tensor *transpose_inputs_kernel[1] = {flattened_kernel_view};
  transpose_kernel(transpose_inputs_kernel, flattened_kernel_transposed, (KernelParams){0, 1});

  u64 grad_im2col_output_shape[2] = {N * H_out * W_out, C * kh * kw};
  u64 *grad_im2col_output_strides = compute_strides(grad_im2col_output_shape, 2);
  Tensor *grad_im2col_output =
      arena_tensor_alloc(a, a, grad_im2col_output_shape, 2, grad_im2col_output_strides,
                         output->dtype, false, NULL, CUDA);
  free(grad_im2col_output_strides);

  Op matmul_op = get_op_impl(MATMUL);
  ForwardKernel matmul_kernel = forward_kernel_dispatcher(matmul_op, CUDA);
  const Tensor *matmul_inputs_grad_input[2] = {flattened_kernel_transposed, output->grad};
  matmul_kernel(matmul_inputs_grad_input, grad_im2col_output, (KernelParams){0, 0});

  // Initialize a_input->grad to zero if it exists
  if (a_input->grad) {
    zeros(a_input->grad, numel(a_input->grad));
  }

  u64 output_elements = N * H_out * W_out * C * kh * kw;
  int block_size = 256;
  int grid_size = (output_elements + block_size - 1) / block_size;

  col2im_cuda_float_kernel<<<grid_size, block_size>>>(
      (const float *)grad_im2col_output->data, (float *)a_input->grad->data, N, C, H_in, W_in, kh,
      kw, stride, a_input->strides[0], a_input->strides[1], a_input->strides[2],
      a_input->strides[3]);
  cudaDeviceSynchronize();

  // 2. Gradient with respect to the kernel (kernel->grad)
  u64 im2col_output_shape[2] = {N * H_out * W_out, C * kh * kw};
  u64 *im2col_output_strides = compute_strides(im2col_output_shape, 2);
  Tensor *im2col_output = arena_tensor_alloc(a, a, im2col_output_shape, 2, im2col_output_strides,
                                             a_input->dtype, false, NULL, CUDA);
  free(im2col_output_strides);

  im2col_cuda_float_kernel<<<grid_size, block_size>>>(
      (const float *)a_input->data, (float *)im2col_output->data, N, C, H_in, W_in, kh, kw, stride,
      a_input->strides[0], a_input->strides[1], a_input->strides[2], a_input->strides[3]);
  cudaDeviceSynchronize();

  Tensor *output_grad_transposed = (Tensor *)arena_alloc(a, sizeof(Tensor), 8);
  memset(output_grad_transposed, 0, sizeof(Tensor));

  const Tensor *transpose_inputs_output_grad[1] = {output->grad};
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

  // Copy data from grad_flattened_kernel to kernel->grad on device
  cudaMemcpy(kernel->grad->data, grad_flattened_kernel->data,
             numel(kernel->grad) * dtype_size(kernel->grad->dtype), cudaMemcpyDeviceToDevice);
}
