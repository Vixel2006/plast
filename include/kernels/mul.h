#pragma once

#include "definitions.h"
#include "tensor.h"
#include <immintrin.h>

#define SIMD_WIDTH 8

void mul_cpu_forward_float_contig_kernel(const float *a, const float *b,
                                         float *c, u64 num_elements);
void mul_cpu_backward_float_contig_kernel(const float *dout, const float *a,
                                          const float *b, float *da, float *db,
                                          u64 num_elements);

void mul_cpu_forward_float_non_contig_kernel(
    const float *a_data, const u64 *a_strides, const float *b_data,
    const u64 *b_strides, float *c_data, const u64 *c_strides, const u64 *shape,
    u64 ndim, u64 num_elements);
void mul_cpu_backward_float_non_contig_kernel(
    const float *dout_data, const u64 *dout_strides, const float *a_data,
    const u64 *a_strides, const float *b_data, const u64 *b_strides,
    float *da_data, const u64 *da_strides, float *db_data,
    const u64 *db_strides, const u64 *shape, u64 ndim, u64 num_elements);

void mul_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params);
void mul_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params);

#ifdef __cplusplus
extern "C" {
#endif

void mul_cuda_forward(const Tensor **inputs, Tensor *output, KernelParams params);
void mul_cuda_backward(Tensor **inputs, const Tensor *output, KernelParams params);

#ifdef __cplusplus
}
#endif
