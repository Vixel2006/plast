#pragma once

#include "core/definitions.h" // Added for u64
#include "core/op.h"
#include "core/tensor.h"
#include <immintrin.h>

#define SIMD_WIDTH 8

void im2col_cpu_float_kernel(float *img, float *buffer, u64 *kernel_size, const u64 *img_shape,
                             const u64 *img_strides, u64 img_ndim, u64 stride);

void col2im_cpu_float_kernel(float *buffer, float *img, u64 *kernel_size, const u64 *img_shape,
                             const u64 *img_strides, u64 img_ndim, u64 stride);

void conv2d_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params);
void conv2d_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params);

void conv_relu_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params);
void conv_relu_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params);

#ifdef __cplusplus
extern "C" {
#endif

void conv2d_cuda_forward(const Tensor **inputs, Tensor *output, KernelParams params);
void conv2d_cuda_backward(Tensor **inputs, const Tensor *output, KernelParams params);

void conv_relu_cuda_forward(const Tensor **inputs, Tensor *output, KernelParams params);
void conv_relu_cuda_backward(Tensor **inputs, const Tensor *output, KernelParams params);

void launch_im2col_cuda_float(const float *img, float *buffer, u64 N, u64 C,
                               u64 H_in, u64 W_in, u64 kh, u64 kw, u64 stride,
                               u64 img_stride_N, u64 img_stride_C,
                               u64 img_stride_H, u64 img_stride_W);
void launch_col2im_cuda_float(const float *buffer, float *img, u64 N, u64 C,
                               u64 H_in, u64 W_in, u64 kh, u64 kw, u64 stride,
                               u64 img_stride_N, u64 img_stride_C,
                               u64 img_stride_H, u64 img_stride_W);

#ifdef __cplusplus
}
#endif
