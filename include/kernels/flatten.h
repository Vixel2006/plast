#pragma once

#include "tensor.h"
#include "definitions.h"

#ifdef __cplusplus
extern "C" {
#endif

void flatten_cpu_forward(const Tensor **inputs, Tensor *output, ...);
void flatten_cpu_backward(Tensor **inputs, const Tensor *output, ...);

#ifdef __cplusplus
}
#endif

