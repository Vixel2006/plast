#pragma once
#include "core/definitions.h"
#include "core/tensor.h"

#define REDUCE_OP_LIST                                                                             \
  REDUCE_OP(SUM, sum)                                                                              \
  REDUCE_OP(MEAN, mean)                                                                            \
  REDUCE_OP(MAX, max)                                                                              \
  REDUCE_OP(MIN, min)

#define REDUCE_OP(NAME, name)                                                                      \
  void name##_cpu_forward(const Tensor **inputs, Tensor *output, KernelParams params);             \
  void name##_cpu_backward(Tensor **inputs, const Tensor *output, KernelParams params);
REDUCE_OP_LIST
#undef REDUCE_OP

#ifdef __cplusplus
extern "C" {
#endif

#define REDUCE_OP(NAME, name)                                                                      \
  void name##_cuda_forward(const Tensor **inputs, Tensor *output, KernelParams params);            \
  void name##_cuda_backward(Tensor **inputs, const Tensor *output, KernelParams params);
REDUCE_OP_LIST
#undef REDUCE_OP

#ifdef __cplusplus
}
#endif
