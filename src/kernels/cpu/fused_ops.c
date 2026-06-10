#include "kernels/matmul.h"
#include "kernels/conv2d.h"
#include "kernels/cpu_utils.h"
#include "kernels/pack.h"
#include "kernels/ops/shape.h"
#include "core/op.h"
#include "core/arena.h"
#include <omp.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#define TILE_SIZE 32

static inline float leaky_relu_fwd(float x, float alpha) {
  return x > 0.0f ? x : x * alpha;
}

static inline float leaky_relu_grad(float x, float alpha) {
  return x > 0.0f ? 1.0f : alpha;
}

// ─── matmul_relu_cpu_forward ────────────────────────────────────────────────

static void matmul_relu_cpu_forward_float_contig_kernel(const float *a, const float *b, float *c,
                                                        u64 batches, u64 rows, u64 inners,
                                                        u64 cols, float alpha) {
#pragma omp parallel for collapse(2) num_threads(8)
  for (u64 batch = 0; batch < batches; ++batch) {
    for (u64 row_tile = 0; row_tile < rows; row_tile += TILE_SIZE) {
      u64 row_tile_end = (row_tile + TILE_SIZE < rows) ? row_tile + TILE_SIZE : rows;
      for (u64 inner_tile = 0; inner_tile < inners; inner_tile += TILE_SIZE) {
        u64 inner_tile_end = (inner_tile + TILE_SIZE < inners) ? inner_tile + TILE_SIZE : inners;
        for (u64 col_tile = 0; col_tile < cols; col_tile += TILE_SIZE) {
          u64 col_tile_end = (col_tile + TILE_SIZE < cols) ? col_tile + TILE_SIZE : cols;
          for (u64 row = row_tile; row < row_tile_end; ++row) {
            for (u64 inner = inner_tile; inner < inner_tile_end; ++inner) {
              for (u64 col = col_tile; col < col_tile_end; ++col) {
                c[batch * rows * cols + row * cols + col] +=
                    a[batch * rows * inners + row * inners + inner] *
                    b[batch * inners * cols + inner * cols + col];
              }
            }
          }
        }
      }
    }
  }

#pragma omp parallel for collapse(3) num_threads(8)
  for (u64 batch = 0; batch < batches; ++batch) {
    for (u64 row = 0; row < rows; ++row) {
      for (u64 col = 0; col < cols; ++col) {
        u64 idx = batch * rows * cols + row * cols + col;
        c[idx] = leaky_relu_fwd(c[idx], alpha);
      }
    }
  }
}

void matmul_relu_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params) {
  const Tensor *a = inputs[0];
  const Tensor *b = inputs[1];
  float alpha = params.fval;

  u64 M = a->shape[a->ndim - 2];
  u64 K = a->shape[a->ndim - 1];
  u64 N = b->shape[b->ndim - 1];

  u64 batches = 1;
  for (u64 i = 0; i < a->ndim - 2; ++i)
    batches *= a->shape[i];

  TensorPack pa, pb;
  tensor_pack_init(&pa, a);
  tensor_pack_init(&pb, b);
  if (!pa.data || !pb.data) {
    tensor_pack_release(&pa);
    tensor_pack_release(&pb);
    return;
  }

  switch (a->dtype) {
  case FLOAT32:
    matmul_relu_cpu_forward_float_contig_kernel((const float *)pa.data, (const float *)pb.data,
                                                (float *)output->data, batches, M, K, N, alpha);
    break;
  default:
    fprintf(stderr, "Unsupported data type for matmul_relu_cpu_forward\n");
    break;
  }

  tensor_pack_release(&pa);
  tensor_pack_release(&pb);
}

// ─── matmul_relu_cpu_backward ───────────────────────────────────────────────

void matmul_cpu_forward_float_nt_kernel(const float *a, const float *b, float *c,
                                        u64 batches, u64 rows, u64 inners, u64 cols);

void matmul_cpu_forward_float_tn_kernel(const float *a, const float *b, float *c,
                                        u64 batches, u64 rows, u64 inners, u64 cols);

static void relu_grad_modulate_contig_kernel(const float *dout, const float *out_data, float *dc,
                                             u64 num_elements, float alpha) {
#pragma omp parallel for num_threads(8)
  for (u64 i = 0; i < num_elements; ++i) {
    dc[i] = dout[i] * leaky_relu_grad(out_data[i], alpha);
  }
}

void matmul_relu_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params) {
  Tensor *a = inputs[0];
  Tensor *b = inputs[1];
  Tensor *da = a->grad;
  Tensor *db = b->grad;
  const Tensor *dc = output->grad;
  float alpha = params.fval;

  if (!dc)
    return;

  u64 M = a->shape[a->ndim - 2];
  u64 K = a->shape[a->ndim - 1];
  u64 N = b->shape[b->ndim - 1];

  u64 batches = 1;
  for (u64 i = 0; i < a->ndim - 2; ++i)
    batches *= a->shape[i];

  u64 num_elements = numel(output);
  float *dc_mod = (float *)malloc(num_elements * sizeof(float));
  if (!dc_mod)
    return;

  relu_grad_modulate_contig_kernel((const float *)dc->data, (const float *)output->data,
                                   dc_mod, num_elements, alpha);

  if (a->requires_grad) {
    TensorPack pb;
    tensor_pack_init(&pb, b);
    if (pb.data) {
      matmul_cpu_forward_float_nt_kernel((const float *)dc_mod, (const float *)pb.data,
                                         (float *)da->data, batches, M, N, K);
    }
    tensor_pack_release(&pb);
  }

  if (b->requires_grad) {
    TensorPack pa;
    tensor_pack_init(&pa, a);
    if (pa.data) {
      matmul_cpu_forward_float_tn_kernel((const float *)pa.data, (const float *)dc_mod,
                                         (float *)db->data, batches, K, M, N);
    }
    tensor_pack_release(&pa);
  }

  free(dc_mod);
}

// ─── matmul_bias_relu_cpu_forward ───────────────────────────────────────────

static void matmul_bias_relu_cpu_forward_float_contig_kernel(const float *a, const float *b,
                                                             const float *bias, float *c,
                                                             u64 batches, u64 rows, u64 inners,
                                                             u64 cols, float alpha) {
#pragma omp parallel for collapse(2) num_threads(8)
  for (u64 batch = 0; batch < batches; ++batch) {
    for (u64 row_tile = 0; row_tile < rows; row_tile += TILE_SIZE) {
      u64 row_tile_end = (row_tile + TILE_SIZE < rows) ? row_tile + TILE_SIZE : rows;
      for (u64 inner_tile = 0; inner_tile < inners; inner_tile += TILE_SIZE) {
        u64 inner_tile_end = (inner_tile + TILE_SIZE < inners) ? inner_tile + TILE_SIZE : inners;
        for (u64 col_tile = 0; col_tile < cols; col_tile += TILE_SIZE) {
          u64 col_tile_end = (col_tile + TILE_SIZE < cols) ? col_tile + TILE_SIZE : cols;
          for (u64 row = row_tile; row < row_tile_end; ++row) {
            for (u64 inner = inner_tile; inner < inner_tile_end; ++inner) {
              for (u64 col = col_tile; col < col_tile_end; ++col) {
                c[batch * rows * cols + row * cols + col] +=
                    a[batch * rows * inners + row * inners + inner] *
                    b[batch * inners * cols + inner * cols + col];
              }
            }
          }
        }
      }
    }
  }

#pragma omp parallel for collapse(3) num_threads(8)
  for (u64 batch = 0; batch < batches; ++batch) {
    for (u64 row = 0; row < rows; ++row) {
      for (u64 col = 0; col < cols; ++col) {
        u64 idx = batch * rows * cols + row * cols + col;
        c[idx] = leaky_relu_fwd(c[idx] + bias[col], alpha);
      }
    }
  }
}

void matmul_bias_relu_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params) {
  const Tensor *a = inputs[0];
  const Tensor *b = inputs[1];
  const Tensor *bias = inputs[2];
  float alpha = params.fval;

  u64 M = a->shape[a->ndim - 2];
  u64 K = a->shape[a->ndim - 1];
  u64 N = b->shape[b->ndim - 1];

  u64 batches = 1;
  for (u64 i = 0; i < a->ndim - 2; ++i)
    batches *= a->shape[i];

  TensorPack pa, pb, pbias;
  tensor_pack_init(&pa, a);
  tensor_pack_init(&pb, b);
  tensor_pack_init(&pbias, bias);
  if (!pa.data || !pb.data || !pbias.data) {
    tensor_pack_release(&pa);
    tensor_pack_release(&pb);
    tensor_pack_release(&pbias);
    return;
  }

  switch (a->dtype) {
  case FLOAT32:
    matmul_bias_relu_cpu_forward_float_contig_kernel(
        (const float *)pa.data, (const float *)pb.data, (const float *)pbias.data,
        (float *)output->data, batches, M, K, N, alpha);
    break;
  default:
    fprintf(stderr, "Unsupported data type for matmul_bias_relu_cpu_forward\n");
    break;
  }

  tensor_pack_release(&pa);
  tensor_pack_release(&pb);
  tensor_pack_release(&pbias);
}

// ─── matmul_bias_relu_cpu_backward ──────────────────────────────────────────

void matmul_bias_relu_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params) {
  Tensor *a = inputs[0];
  Tensor *b = inputs[1];
  Tensor *bias = inputs[2];
  Tensor *da = a->grad;
  Tensor *db = b->grad;
  Tensor *dbias = bias->grad;
  const Tensor *dc = output->grad;
  float alpha = params.fval;

  if (!dc)
    return;

  u64 M = a->shape[a->ndim - 2];
  u64 K = a->shape[a->ndim - 1];
  u64 N = b->shape[b->ndim - 1];

  u64 batches = 1;
  for (u64 i = 0; i < a->ndim - 2; ++i)
    batches *= a->shape[i];

  u64 num_elements = numel(output);
  float *dc_mod = (float *)malloc(num_elements * sizeof(float));
  if (!dc_mod)
    return;

  relu_grad_modulate_contig_kernel((const float *)dc->data, (const float *)output->data,
                                   dc_mod, num_elements, alpha);

  if (a->requires_grad) {
    TensorPack pb;
    tensor_pack_init(&pb, b);
    if (pb.data) {
      matmul_cpu_forward_float_nt_kernel((const float *)dc_mod, (const float *)pb.data,
                                         (float *)da->data, batches, M, N, K);
    }
    tensor_pack_release(&pb);
  }

  if (b->requires_grad) {
    TensorPack pa;
    tensor_pack_init(&pa, a);
    if (pa.data) {
      matmul_cpu_forward_float_tn_kernel((const float *)pa.data, (const float *)dc_mod,
                                         (float *)db->data, batches, K, M, N);
    }
    tensor_pack_release(&pa);
  }

  if (bias->requires_grad && dbias) {
    float *dbias_data = (float *)dbias->data;
    memset(dbias_data, 0, N * sizeof(float));
    for (u64 batch = 0; batch < batches; ++batch) {
      for (u64 row = 0; row < M; ++row) {
        for (u64 col = 0; col < N; ++col) {
          u64 idx = batch * M * N + row * N + col;
          dbias_data[col] += dc_mod[idx];
        }
      }
    }
  }

  free(dc_mod);
}

// ─── conv_relu_cpu_forward ──────────────────────────────────────────────────

void conv_relu_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params) {
  Arena *a = (Arena *)params.dim;
  u64 stride = params.keepdim;
  float alpha = params.fval;

  const Tensor *a_input = inputs[0];
  const Tensor *kernel = inputs[1];

  Tensor *flattened_kernel_view = (Tensor *)arena_alloc(a, sizeof(Tensor), 8);
  memset(flattened_kernel_view, 0, sizeof(Tensor));
  flattened_kernel_view->ndim = 2;
  flattened_kernel_view->shape[0] = kernel->shape[0];
  flattened_kernel_view->shape[1] = kernel->shape[1] * kernel->shape[2] * kernel->shape[3];

  Op flatten_op = get_op_impl(FLATTEN);
  ForwardKernel flatten_kernel = forward_kernel_dispatcher(flatten_op, CPU);
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
                                             a_input->dtype, a_input->requires_grad, NULL, CPU);
  free(im2col_output_strides);

  u64 kernel_size_arr[2] = {kh, kw};
  im2col_cpu_float_kernel((float *)a_input->data, (float *)im2col_output->data, kernel_size_arr,
                          a_input->shape, a_input->strides, a_input->ndim, stride);

  Tensor *flattened_kernel_transposed = (Tensor *)arena_alloc(a, sizeof(Tensor), 8);
  memset(flattened_kernel_transposed, 0, sizeof(Tensor));

  Op transpose_op = get_op_impl(TRANSPOSE);
  ForwardKernel transpose_kernel = forward_kernel_dispatcher(transpose_op, CPU);
  const Tensor *transpose_inputs[1] = {flattened_kernel_view};
  transpose_kernel(transpose_inputs, flattened_kernel_transposed, (KernelParams){0, 1});

  // matmul with inline relu via our fused kernel params
  Op mm_relu_op = get_op_impl(MATMUL_RELU);
  ForwardKernel mm_relu_kernel = forward_kernel_dispatcher(mm_relu_op, CPU);
  const Tensor *mm_inputs[2] = {im2col_output, flattened_kernel_transposed};
  KernelParams mm_params = {.dim = 0, .keepdim = 0, .fval = alpha};
  mm_relu_kernel(mm_inputs, output, mm_params);
}

// ─── conv_relu_cpu_backward ─────────────────────────────────────────────────

void conv_relu_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params) {
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

  // Modulate upstream gradient by relu gradient
  u64 numel_output = numel(output);
  float *dc_mod = (float *)malloc(numel_output * sizeof(float));
  if (!dc_mod)
    return;
  relu_grad_modulate_contig_kernel((const float *)output->grad->data,
                                   (const float *)output->data, dc_mod, numel_output, alpha);

  Tensor *flattened_kernel_view = (Tensor *)arena_alloc(a, sizeof(Tensor), 8);
  memset(flattened_kernel_view, 0, sizeof(Tensor));
  flattened_kernel_view->ndim = 2;
  flattened_kernel_view->shape[0] = kernel->shape[0];
  flattened_kernel_view->shape[1] = kernel->shape[1] * kernel->shape[2] * kernel->shape[3];

  Op flatten_op = get_op_impl(FLATTEN);
  ForwardKernel flatten_kernel = forward_kernel_dispatcher(flatten_op, CPU);
  const Tensor *flatten_inputs[1] = {kernel};
  flatten_kernel(flatten_inputs, flattened_kernel_view, (KernelParams){0, 0});

  // Gradient w.r.t. input
  u64 grad_im2col_output_shape[2] = {N * H_out * W_out, C * kh * kw};
  u64 *grad_im2col_output_strides = compute_strides(grad_im2col_output_shape, 2);
  Tensor *grad_im2col_output = arena_tensor_alloc(
      a, a, grad_im2col_output_shape, 2, grad_im2col_output_strides, output->dtype, false, NULL, CPU);
  free(grad_im2col_output_strides);

  // Wrap dc_mod as a temporary Tensor for the matmul call
  Tensor dc_mod_tensor;
  memset(&dc_mod_tensor, 0, sizeof(Tensor));
  dc_mod_tensor.data = dc_mod;
  dc_mod_tensor.ndim = output->ndim;
  memcpy(dc_mod_tensor.shape, output->shape, output->ndim * sizeof(u64));
  memcpy(dc_mod_tensor.strides, output->strides, output->ndim * sizeof(u64));
  dc_mod_tensor.dtype = output->dtype;
  dc_mod_tensor.device = CPU;

  Op matmul_op = get_op_impl(MATMUL);
  ForwardKernel matmul_kernel = forward_kernel_dispatcher(matmul_op, CPU);
  const Tensor *matmul_inputs_grad_input[2] = {&dc_mod_tensor, flattened_kernel_view};
  matmul_kernel(matmul_inputs_grad_input, grad_im2col_output, (KernelParams){0, 0});

  if (a_input->grad) {
    memset(a_input->grad->data, 0, numel(a_input->grad) * dtype_size(a_input->grad->dtype));
  }

  u64 kernel_size_arr[2] = {kh, kw};
  col2im_cpu_float_kernel((float *)grad_im2col_output->data, (float *)a_input->grad->data,
                          kernel_size_arr, a_input->shape, a_input->strides, a_input->ndim, stride);

  // Gradient w.r.t. kernel
  u64 im2col_output_shape[2] = {N * H_out * W_out, C * kh * kw};
  u64 *im2col_output_strides = compute_strides(im2col_output_shape, 2);
  Tensor *im2col_output =
      arena_tensor_alloc(a, a, im2col_output_shape, 2, im2col_output_strides, a_input->dtype, false,
                         NULL, CPU);
  free(im2col_output_strides);

  im2col_cpu_float_kernel((float *)a_input->data, (float *)im2col_output->data, kernel_size_arr,
                          a_input->shape, a_input->strides, a_input->ndim, stride);

  Tensor *output_grad_transposed = (Tensor *)arena_alloc(a, sizeof(Tensor), 8);
  memset(output_grad_transposed, 0, sizeof(Tensor));

  Op transpose_op = get_op_impl(TRANSPOSE);
  ForwardKernel transpose_kernel = forward_kernel_dispatcher(transpose_op, CPU);
  const Tensor *transpose_inputs_output_grad[1] = {&dc_mod_tensor};
  transpose_kernel(transpose_inputs_output_grad, output_grad_transposed, (KernelParams){0, 1});

  u64 grad_flattened_kernel_shape[2] = {kernel->shape[0],
                                        kernel->shape[1] * kernel->shape[2] * kernel->shape[3]};
  u64 *grad_flattened_kernel_strides = compute_strides(grad_flattened_kernel_shape, 2);
  Tensor *grad_flattened_kernel =
      arena_tensor_alloc(a, a, grad_flattened_kernel_shape, 2, grad_flattened_kernel_strides,
                         output->dtype, false, NULL, CPU);
  free(grad_flattened_kernel_strides);

  const Tensor *matmul_inputs_grad_kernel[2] = {output_grad_transposed, im2col_output};
  matmul_kernel(matmul_inputs_grad_kernel, grad_flattened_kernel, (KernelParams){0, 0});

  if (kernel->grad) {
    memset(kernel->grad->data, 0, numel(kernel->grad) * dtype_size(kernel->grad->dtype));
  }

  memcpy(kernel->grad->data, grad_flattened_kernel->data,
         numel(kernel->grad) * dtype_size(kernel->grad->dtype));

  free(dc_mod);
}
