#pragma once

#include "op.h"
#include "tensor.h"
#include <immintrin.h>

#define SIMD_WIDTH 8

void matmul_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params);
void matmul_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params);

#ifdef __cplusplus
extern "C" {
#endif

void matmul_cuda_forward(const Tensor **inputs, Tensor *output, KernelParams params);
void matmul_cuda_backward(Tensor **inputs, const Tensor *output, KernelParams params);

#ifdef __cplusplus
}
#endif
