#ifndef OPTIMIZERS_ZERO_GRAD_H
#define OPTIMIZERS_ZERO_GRAD_H

#include "core/tensor.h"

void zero_grad_cpu(Tensor *t);

#ifdef __cplusplus
extern "C" {
#endif
void zero_grad_cuda(Tensor *t);
#ifdef __cplusplus
}
#endif

#endif // OPTIMIZERS_ZERO_GRAD_H