#pragma once

#include "tensor.h"
#include "definitions.h"

void view_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params);
void view_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params);
