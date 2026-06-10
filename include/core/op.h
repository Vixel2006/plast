#pragma once

#include "core/tensor.h"

typedef void (*ForwardKernel)(const Tensor **inputs, Tensor *output, KernelParams params);
typedef void (*BackwardKernel)(Tensor **inputs, const Tensor *output, KernelParams params);

typedef enum OP_TYPE {
  ADD,
  SUB,
  MUL,
  DIV,
  MATMUL,
  LEAKY_RELU,
  LOG,
  EXP,
  ABS,
  NEG,
  SIN,
  COS,
  TAN,
  VIEW,
  TRANSPOSE,
  UNSQUEEZE,
  SQUEEZE,
  EXPAND,
  FLATTEN,
  BROADCAST,
  MEAN,
  MIN,
  MAX,
  SUM,
  CONV2D,
  MATMUL_RELU,
  MATMUL_BIAS_RELU,
  CONV_RELU
} OP_TYPE;

typedef struct Op {
  ForwardKernel cpu_forward;
  ForwardKernel cuda_forward;
  BackwardKernel cpu_backward;
  BackwardKernel cuda_backward;
} Op;

#ifdef __cplusplus
extern "C" {
#endif

Op get_op_impl(OP_TYPE op_type);

ForwardKernel forward_kernel_dispatcher(Op op, DEVICE device);
BackwardKernel backward_kernel_dispatcher(Op op, DEVICE device);

#ifdef __cplusplus
}
#endif
