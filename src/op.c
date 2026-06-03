#include "op.h"
#include "kernels/abs.h"
#include "kernels/add.h"
#include "kernels/broadcast.h"
#include "kernels/conv2d.h"
#include "kernels/cos.h"
#include "kernels/div.h"
#include "kernels/exp.h"
#include "kernels/expand.h"
#include "kernels/flatten.h"
#include "kernels/leaky_relu.h"
#include "kernels/log.h"
#include "kernels/matmul.h"
#include "kernels/max.h"
#include "kernels/mean.h"
#include "kernels/min.h"
#include "kernels/mul.h"
#include "kernels/neg.h"
#include "kernels/sin.h"
#include "kernels/squeeze.h"
#include "kernels/sub.h"
#include "kernels/sum.h"
#include "kernels/tan.h"
#include "kernels/transpose.h"
#include "kernels/unsqueeze.h"
#include "kernels/view.h"

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
#include "ops.def"
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
