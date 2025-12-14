#include "plast/core/shape_utils_c.h"
#include "plast/kernels/cpu/unary_kernels.h"

#include <immintrin.h>
#include <math.h>
#include <sleef.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define SIMD_WIDTH 8

void plast_cpu_exp_kernel_float(float* out, const float* in, size_t num_elements)
{
    size_t i = 0;

    for (; i + SIMD_WIDTH - 1 < num_elements; i += SIMD_WIDTH)
    {
        __m256 x = _mm256_loadu_ps(in + i);
        __m256 y = Sleef_expf8_u10avx2(x);
        _mm256_storeu_ps(out + i, y);
    }

    for (; i < num_elements; ++i)
    {
        out[i] = expf(in[i]);
    }
}



void plast_cpu_exp_kernel_int32(int32_t* out, const int32_t* in, size_t num_elements)
{
    for (size_t i = 0; i < num_elements; ++i)
    {
        out[i] = (int32_t) expf((float) in[i]);
    }
}
