#pragma once

#include "core/tensor.h"
#include "core/definitions.h"

void view_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params);
void view_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params);
