#include "op.h"
#include "kernels/abs.h"
#include "kernels/add.h"
#include "kernels/broadcast.h"
#include "kernels/cos.h"
#include "kernels/div.h"
#include "kernels/exp.h"
#include "kernels/expand.h"
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

Op get_op_impl(OP_TYPE op_type) {
  Op op;
  switch (op_type) {
  case ADD:
    op.cpu_forward = add_cpu_forward;
    op.cpu_backward = add_cpu_backward;
    op.cuda_forward = add_cuda_forward;
    op.cuda_backward = add_cuda_backward;
    break;
  case SUB:
    op.cpu_forward = sub_cpu_forward;
    op.cpu_backward = sub_cpu_backward;
    op.cuda_forward = sub_cuda_forward;
    op.cuda_backward = sub_cuda_backward;
    break;
  case MUL:
    op.cpu_forward = mul_cpu_forward;
    op.cpu_backward = mul_cpu_backward;
    op.cuda_forward = mul_cuda_forward;
    op.cuda_backward = mul_cuda_backward;
    break;
  case DIV:
    op.cpu_forward = div_cpu_forward;
    op.cpu_backward = div_cpu_backward;
    op.cuda_forward = div_cuda_forward;
    op.cuda_backward = div_cuda_backward;
    break;
  case MATMUL:
    op.cpu_forward = matmul_cpu_forward;
    op.cpu_backward = matmul_cpu_backward;
    op.cuda_forward = matmul_cuda_forward;
    op.cuda_backward = matmul_cuda_backward;
    break;
  case LEAKY_RELU:
    op.cpu_forward = leaky_relu_cpu_forward;
    op.cpu_backward = leaky_relu_cpu_backward;
    op.cuda_forward = leaky_relu_cuda_forward;
    op.cuda_backward = leaky_relu_cuda_backward;
    break;
  case LOG:
    op.cpu_forward = log_cpu_forward;
    op.cpu_backward = log_cpu_backward;
    op.cuda_forward = log_cuda_forward;
    op.cuda_backward = log_cuda_backward;
    break;
  case EXP:
    op.cpu_forward = exp_cpu_forward;
    op.cpu_backward = exp_cpu_backward;
    op.cuda_forward = exp_cuda_forward;
    op.cuda_backward = exp_cuda_backward;
    break;
  case ABS:
    op.cpu_forward = abs_cpu_forward;
    op.cpu_backward = abs_cpu_backward;
    op.cuda_forward = abs_cuda_forward;
    op.cuda_backward = abs_cuda_backward;
    break;
  case NEG:
    op.cpu_forward = neg_cpu_forward;
    op.cpu_backward = neg_cpu_backward;
    op.cuda_forward = neg_cuda_forward;
    op.cuda_backward = neg_cuda_backward;
    break;
  case SIN:
    op.cpu_forward = sin_cpu_forward;
    op.cpu_backward = sin_cpu_backward;
    op.cuda_forward = sin_cuda_forward;
    op.cuda_backward = sin_cuda_backward;
    break;
  case COS:
    op.cpu_forward = cos_cpu_forward;
    op.cpu_backward = cos_cpu_backward;
    op.cuda_forward = cos_cuda_forward;
    op.cuda_backward = cos_cuda_backward;
    break;
  case TAN:
    op.cpu_forward = tan_cpu_forward;
    op.cpu_backward = tan_cpu_backward;
    op.cuda_forward = tan_cuda_forward;
    op.cuda_backward = tan_cuda_backward;
    break;
  case VIEW:
    op.cpu_forward = view_cpu_forward;
    op.cpu_backward = view_cpu_backward;
    op.cuda_forward = view_cpu_forward;
    op.cuda_backward = view_cpu_backward;
    break;
  case TRANSPOSE:
    op.cpu_forward = transpose_cpu_forward;
    op.cpu_backward = transpose_cpu_backward;
    op.cuda_forward = transpose_cpu_forward;
    op.cuda_backward = transpose_cpu_backward;
    break;
  case UNSQUEEZE:
    op.cpu_forward = unsqueeze_cpu_forward;
    op.cpu_backward = unsqueeze_cpu_backward;
    op.cuda_forward = unsqueeze_cpu_forward;
    op.cuda_backward = unsqueeze_cpu_backward;
    break;
  case SQUEEZE:
    op.cpu_forward = squeeze_cpu_forward;
    op.cpu_backward = squeeze_cpu_backward;
    op.cuda_forward = squeeze_cpu_forward;
    op.cuda_backward = squeeze_cpu_backward;
    break;
  case EXPAND:
    op.cpu_forward = expand_cpu_forward;
    op.cpu_backward = expand_cpu_backward;
    op.cuda_forward = expand_cpu_forward;
    op.cuda_backward = expand_cpu_backward;
    break;
  case BROADCAST:
    op.cpu_forward = broadcast_cpu_forward;
    op.cpu_backward = broadcast_cpu_backward;
    op.cuda_forward = broadcast_cpu_forward;
    op.cuda_backward = broadcast_cpu_backward;
    break;
  case MEAN:
    op.cpu_forward = mean_cpu_forward;
    op.cpu_backward = mean_cpu_backward;
    op.cuda_forward = mean_cuda_forward;
    op.cuda_backward = mean_cuda_backward;
    break;
  case MIN:
    op.cpu_forward = min_cpu_forward;
    op.cpu_backward = min_cpu_backward;
    op.cuda_forward = min_cuda_forward;
    op.cuda_backward = min_cuda_backward;
    break;
  case MAX:
    op.cpu_forward = max_cpu_forward;
    op.cpu_backward = max_cpu_backward;
    op.cuda_forward = max_cuda_forward;
    op.cuda_backward = max_cuda_backward;
    break;
  case SUM:
    op.cpu_forward = sum_cpu_forward;
    op.cpu_backward = sum_cpu_backward;
    op.cuda_forward = sum_cuda_forward;
    op.cuda_backward = sum_cuda_backward;
    break;
  default:
    op.cpu_forward = NULL;
    op.cpu_backward = NULL;
    op.cuda_forward = NULL;
    op.cuda_backward = NULL;
    break;
  }
  return op;
}

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
