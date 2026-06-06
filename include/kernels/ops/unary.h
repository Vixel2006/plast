#pragma once
#include "core/definitions.h"
#include "core/tensor.h"

#define UNARY_OP_LIST                                                                              \
  UNARY_OP(ABS, abs)                                                                               \
  UNARY_OP(COS, cos)                                                                               \
  UNARY_OP(SIN, sin)                                                                               \
  UNARY_OP(TAN, tan)                                                                               \
  UNARY_OP(EXP, exp)                                                                               \
  UNARY_OP(LOG, log)                                                                               \
  UNARY_OP(NEG, neg)                                                                               \
  UNARY_OP(LEAKY_RELU, leaky_relu)

#define UNARY_OP(NAME, name)                                                                       \
  void name##_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params);             \
  void name##_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params);
UNARY_OP_LIST
#undef UNARY_OP

#ifdef __cplusplus
extern "C" {
#endif

#define UNARY_OP(NAME, name)                                                                       \
  void name##_cuda_forward(const Tensor **inputs, Tensor *output, KernelParams params);            \
  void name##_cuda_backward(Tensor **inputs, const Tensor *output, KernelParams params);
UNARY_OP_LIST
#undef UNARY_OP

#ifdef __cplusplus
}
#endif
