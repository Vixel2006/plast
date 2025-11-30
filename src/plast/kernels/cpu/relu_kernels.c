#include "plast/core/shape_utils_c.h" // For get_index, increment_indices
#include "plast/kernels/cpu/unary_kernels.h"

#include <immintrin.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define SIMD_WIDTH 8

void plast_cpu_relu_kernel_float(float* out, const float* in, size_t num_elements)
{
    size_t i = 0;

    for (; i + SIMD_WIDTH - 1 < num_elements; i += SIMD_WIDTH)
    {
        __m256 x = _mm256_loadu_ps(in + i);
        __m256 zero = _mm256_setzero_ps();
        __m256 y = _mm256_max_ps(x, zero);
        _mm256_storeu_ps(out + i, y);
    }

    for (; i < num_elements; ++i)
    {
        out[i] = fmaxf(0.0f, in[i]);
    }
}

void plast_cpu_relu_kernel_strided_float(float* out, const float* in, const size_t* out_shape,
                                         size_t out_ndim, const size_t* in_strides)
{
    size_t total_elements = 1;
    for (size_t i = 0; i < out_ndim; ++i)
    {
        total_elements *= out_shape[i];
    }

    size_t* current_indices = (size_t*) calloc(out_ndim, sizeof(size_t));
    if (!current_indices)
    {
        // Handle allocation error
        return;
    }

    for (size_t i = 0; i < total_elements; ++i)
    {
        size_t in_idx = get_index(current_indices, in_strides, out_ndim);
        out[i] = fmaxf(0.0f, in[in_idx]);

        increment_indices(current_indices, out_shape, out_ndim);
    }
    free(current_indices);
}
