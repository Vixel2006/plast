#pragma once

#include "core/op.h"
#include "core/tensor.h"
#include <immintrin.h>

#define SIMD_WIDTH 8

void add_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params);
void add_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params);

#ifdef __cplusplus
extern "C" {
#endif

void add_cuda_forward(const Tensor **inputs, Tensor *output, KernelParams params);
void add_cuda_backward(Tensor **inputs, const Tensor *output, KernelParams params);

#ifdef __cplusplus
}
#endif
