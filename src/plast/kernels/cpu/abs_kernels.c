#include "plast/core/shape_utils_c.h"
#include "plast/kernels/cpu/unary_kernels.h"

#include <immintrin.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define SIMD_WIDTH 8

void plast_cpu_abs_kernel_float(float* out, const float* in, size_t num_elements)
{
    size_t i = 0;

    for (; i + SIMD_WIDTH - 1 < num_elements; i += SIMD_WIDTH)
    {
        __m256 x = _mm256_loadu_ps(in + i);

        // NOTE: Abs can be done by applying a mask that has 0 in the sign bit, then by applying the
        // logical and on the number we get the absolute value
        // http://steve.hollasch.net/cgindex/coding/ieeefloat.html for understanding floating-point
        // layout

        __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
        __m256 y = _mm256_and_ps(x, abs_mask);

        _mm256_storeu_ps(out + i, y);
    }

    for (; i < num_elements; ++i)
    {
        out[i] = fabsf(in[i]);
    }
}

void plast_cpu_abs_kernel_int32(int32_t* out, const int32_t* in, size_t num_elements)
{
    for (size_t i = 0; i < num_elements; ++i)
    {
        out[i] = abs(in[i]);
    }
}

