#include "plast/core/shape_utils_c.h"
#include "plast/kernels/cpu/binary_kernels.h"
#include <immintrin.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define SIMD_WIDTH 8

void plast_cpu_sub_kernel_float(float* out, const float* in1, const float* in2, size_t num_elements)
{
    size_t i = 0;
    for (; i + SIMD_WIDTH - 1 < num_elements; i += SIMD_WIDTH)
    {
        __m256 x = _mm256_loadu_ps(in1 + i);
        __m256 y = _mm256_loadu_ps(in2 + i);
        __m256 z = _mm256_sub_ps(x, y);
        _mm256_storeu_ps(out + i, z);
    }

    for (; i < num_elements; ++i)
    {
        out[i] = in1[i] - in2[i];
    }
}

void plast_cpu_sub_kernel_int32(int32_t* out, const int32_t* in1, const int32_t* in2,
                                size_t num_elements)
{
    for (size_t i = 0; i < num_elements; ++i)
    {
        out[i] = in1[i] - in2[i];
    }
}

