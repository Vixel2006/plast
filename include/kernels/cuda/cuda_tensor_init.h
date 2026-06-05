#pragma once

#include "core/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

void zeros_cuda(Tensor *t, u64 num_elements);
void ones_cuda(Tensor *t, u64 num_elements);
void set_ones_grad_cuda(Tensor *t);

#ifdef __cplusplus
}
#endif
