#pragma once

#include "definitions.h"
#include "tensor.h"
#include <immintrin.h>

#define SIMD_WIDTH 8

void cos_cpu_forward_float_contig_kernel(const float *a, float *c, u64 num_elements);
void cos_cpu_backward_float_contig_kernel(const float *dout, const float *a, float *da,
                                          u64 num_elements);

void cos_cpu_forward_float_non_contig_kernel(const float *a_data, const u64 *a_strides,
                                             float *c_data, const u64 *c_strides, const u64 *shape,
                                             u64 ndim, u64 num_elements);
void cos_cpu_backward_float_non_contig_kernel(const float *dout_data, const u64 *dout_strides,
                                              const float *a_data, const u64 *a_strides,
                                              float *da_data, const u64 *da_strides,
                                              const u64 *shape, u64 ndim, u64 num_elements);

void cos_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params);
void cos_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params);

#ifdef __cplusplus
extern "C" {
#endif

void cos_cuda_forward(const Tensor **inputs, Tensor *output, KernelParams params);
void cos_cuda_backward(Tensor **inputs, const Tensor *output, KernelParams params);

#ifdef __cplusplus
}
#endif
