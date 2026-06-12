#pragma once

#include "core/op.h"
#include "core/tensor.h"
#include <immintrin.h>

#define SIMD_WIDTH 8

void matmul_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params);
void matmul_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params);

void matmul_relu_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params);
void matmul_relu_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params);

void matmul_bias_relu_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params);
void matmul_bias_relu_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params);

#ifdef __cplusplus
extern "C" {
#endif

void matmul_cuda_forward(const Tensor **inputs, Tensor *output, KernelParams params);
void matmul_cuda_backward(Tensor **inputs, const Tensor *output, KernelParams params);

void matmul_relu_cuda_forward(const Tensor **inputs, Tensor *output, KernelParams params);
void matmul_relu_cuda_backward(Tensor **inputs, const Tensor *output, KernelParams params);

void matmul_bias_relu_cuda_forward(const Tensor **inputs, Tensor *output, KernelParams params);
void matmul_bias_relu_cuda_backward(Tensor **inputs, const Tensor *output, KernelParams params);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <cuda_runtime.h>

extern "C" {
void launch_matmul_nt_cuda(const float *a, const float *b, float *c, u64 batches, u64 rows,
                           u64 inners, u64 cols, dim3 grid_dim, dim3 block_dim);
void launch_matmul_tn_cuda(const float *a, const float *b, float *c, u64 batches, u64 rows,
                           u64 inners, u64 cols, dim3 grid_dim, dim3 block_dim);
void launch_relu_grad_modulate_cuda(const float *dout, const float *out_data, float *dc,
                                    u64 num_elements, float alpha, int grid_size, int block_size);
}
#endif
