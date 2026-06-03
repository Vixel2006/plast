#pragma once

#include "definitions.h"
#include "tensor.h"

void expand_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params);
void expand_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params);
