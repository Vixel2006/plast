#pragma once

#include "tensor.h"
#include "definitions.h"

#ifdef __cplusplus
extern "C" {
#endif

void flatten_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params);
void flatten_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params);

#ifdef __cplusplus
}
#endif

