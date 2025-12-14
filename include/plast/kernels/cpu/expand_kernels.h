#pragma once

#include "plast/core/types.h"
#include <stddef.h> // For size_t

#ifdef __cplusplus
extern "C" {
#endif

void cpu_expand_kernel(const void* input_data,
                       const size_t* input_shape,
                       const size_t* input_strides,
                       size_t input_ndim,
                       void* output_data,
                       const size_t* output_shape,
                       size_t output_ndim,
                       size_t item_size);

#ifdef __cplusplus
}
#endif
