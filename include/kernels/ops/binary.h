#pragma once
#include "core/definitions.h"
#include "core/tensor.h"

#define BINARY_OP_LIST                                                                             \
  BINARY_OP(ADD, add)                                                                              \
  BINARY_OP(SUB, sub)                                                                              \
  BINARY_OP(MUL, mul)                                                                              \
  BINARY_OP(DIV, div)

#define BINARY_OP(NAME, name)                                                                      \
  void name##_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params);             \
  void name##_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params);
BINARY_OP_LIST
#undef BINARY_OP

#ifdef __cplusplus
extern "C" {
#endif

#define BINARY_OP(NAME, name)                                                                      \
  void name##_cuda_forward(const Tensor **inputs, Tensor *output, KernelParams params);            \
  void name##_cuda_backward(Tensor **inputs, const Tensor *output, KernelParams params);
BINARY_OP_LIST
#undef BINARY_OP

#ifdef __cplusplus
}
#endif
