#pragma once
#include "core/definitions.h"
#include "core/tensor.h"

void view_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params);
void view_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params);

void flatten_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params);
void flatten_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params);

void squeeze_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params);
void squeeze_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params);

void unsqueeze_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params);
void unsqueeze_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params);

void expand_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params);
void expand_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params);

void broadcast_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params);
void broadcast_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params);

void transpose_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params);
void transpose_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params);
