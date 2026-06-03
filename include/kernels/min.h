#pragma once

#include "tensor.h"
#include <stdarg.h>

void min_cpu_forward_float_contig_kernel(const float *a, float *c,
                                         u64 num_elements);
void min_cpu_backward_float_contig_kernel(const float *a, const float *c, const float *dc, float *da, u64 num_elements);

void min_cpu_forward_float_non_contig_kernel(const float *a_data,
                                             const u64 *a_strides,
                                                                                          const u64 *shape, u64 ndim,
                                             u64 num_elements, float *c_data);
void min_cpu_backward_float_non_contig_kernel(const float *a_data, const u64 *a_strides, const float *c_data, const float *dc_data, float *da_data, const u64 *da_strides, const u64 *shape, u64 ndim, u64 num_elements);

void min_cpu_forward_float_dim_kernel(const float *a_data, const u64 *a_strides,
                                      const u64 *a_shape, u64 a_ndim,
                                      float *c_data, const u64 *c_strides,
                                      const u64 *c_shape, u64 c_ndim, u64 dim,
                                      bool keepdim);
void min_cpu_backward_float_dim_kernel(const float *a_data,
                                       const u64 *a_strides, const u64 *a_shape,
                                       u64 a_ndim, const float *c_data,
                                       const u64 *c_strides, const u64 *c_shape,
                                       u64 c_ndim, const float *dc_data,
                                       float *da_data, const u64 *da_strides,
                                       u64 dim, bool keepdim);

void min_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params);
void min_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params);

#ifdef __cplusplus
extern "C" {
#endif

void min_cuda_forward(const Tensor **inputs, Tensor *output, KernelParams params);
void min_cuda_backward(Tensor **inputs, const Tensor *output, KernelParams params);

#ifdef __cplusplus
}
#endif
