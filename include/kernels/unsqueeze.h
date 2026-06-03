#pragma once

#include "tensor.h"
#include "definitions.h"

void unsqueeze_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params);
void unsqueeze_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params);
