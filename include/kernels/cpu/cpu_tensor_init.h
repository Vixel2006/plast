#pragma once

#include "core/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

void zeros_cpu(Tensor *t, u64 num_elements);
void ones_cpu(Tensor *t, u64 num_elements);
void set_ones_grad_cpu(Tensor *t);

#ifdef __cplusplus
}
#endif
