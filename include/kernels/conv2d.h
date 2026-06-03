#pragma once

#include "definitions.h" // Added for u64
#include "op.h"
#include "tensor.h"
#include <immintrin.h>

#define SIMD_WIDTH 8

void im2col_cpu_float_kernel(float *img, float *buffer, u64 *kernel_size, const u64 *img_shape,
                             const u64 *img_strides, u64 img_ndim, u64 stride);

void col2im_cpu_float_kernel(float *buffer, float *img, u64 *kernel_size, const u64 *img_shape,
                             const u64 *img_strides, u64 img_ndim, u64 stride);

void conv2d_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params);
void conv2d_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params);

#ifdef __cplusplus
extern "C" {
#endif

void conv2d_cuda_forward(const Tensor **inputs, Tensor *output, KernelParams params);
void conv2d_cuda_backward(Tensor **inputs, const Tensor *output, KernelParams params);

#ifdef __cplusplus
}
#endif
