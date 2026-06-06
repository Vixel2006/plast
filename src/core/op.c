#include "core/op.h"
#include "kernels/ops/unary.h"
#include "kernels/ops/binary.h"
#include "kernels/ops/reduce.h"
#include "kernels/ops/shape.h"
#include "kernels/matmul.h"
#include "kernels/conv2d.h"

// Op has a CUDA implementation — assign both backends
#ifdef CUDA_AVAILABLE
#define OP_CUDA(NAME, name)                                                                        \
  case NAME:                                                                                       \
    op.cpu_forward = name##_cpu_forward;                                                           \
    op.cpu_backward = name##_cpu_backward;                                                         \
    op.cuda_forward = name##_cuda_forward;                                                         \
    op.cuda_backward = name##_cuda_backward;                                                       \
    break;
#else
#define OP_CUDA(NAME, name)                                                                        \
  case NAME:                                                                                       \
    op.cpu_forward = name##_cpu_forward;                                                           \
    op.cpu_backward = name##_cpu_backward;                                                         \
    op.cuda_forward = NULL;                                                                        \
    op.cuda_backward = NULL;                                                                       \
    break;
#endif

// CPU-only op — CUDA fallback uses same function (shape ops, etc.)
#define OP_CPU(NAME, name)                                                                         \
  case NAME:                                                                                       \
    op.cpu_forward = name##_cpu_forward;                                                           \
    op.cpu_backward = name##_cpu_backward;                                                         \
    op.cuda_forward = name##_cpu_forward;                                                          \
    op.cuda_backward = name##_cpu_backward;                                                        \
    break;

Op get_op_impl(OP_TYPE op_type) {
  Op op;
  switch (op_type) {
#include "core/ops.def"
  default:
    op.cpu_forward = NULL;
    op.cpu_backward = NULL;
    op.cuda_forward = NULL;
    op.cuda_backward = NULL;
    break;
  }
  return op;
}

#undef OP_CUDA
#undef OP_CPU

ForwardKernel forward_kernel_dispatcher(Op op, DEVICE device) {
  switch (device) {
  case CPU:
    return op.cpu_forward;
  case CUDA:
    return op.cuda_forward;
  default:
    return NULL;
  }
}

BackwardKernel backward_kernel_dispatcher(Op op, DEVICE device) {
  switch (device) {
  case CPU:
    return op.cpu_backward;
  case CUDA:
    return op.cuda_backward;
  default:
    return NULL;
  }
}
