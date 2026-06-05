#pragma once

#include "core/tensor.h"
#include "core/definitions.h"

#ifdef __cplusplus
extern "C" {
#endif

void transpose_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params);
void transpose_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params);

#ifdef __cplusplus
}
#endif
