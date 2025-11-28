#include "plast/kernels/cpu/binary_kernels.h" // For function declarations
#include <immintrin.h>                        // For SIMD instructions
#include <math.h>                             // For general math functions
#include <stddef.h>                           // For size_t
#include <stdint.h>                           // For int32_t
#include <string.h>                           // For memcpy (if needed)

#define SIMD_WIDTH 8

void plast_cpu_mul_kernel_float(float* out, const float* in1, const float* in2, size_t num_elements)
{
    size_t i = 0;

#ifdef __AVX2__
    for (; i + SIMD_WIDTH - 1 < num_elements; i += SIMD_WIDTH)
    {
        __m256 x = _mm256_loadu_ps(in1 + i);
        __m256 y = _mm256_loadu_ps(in2 + i);
        __m256 z = _mm256_mul_ps(x, y);
        _mm256_store_ps(out + i, z);
    }
#endif

    for (; i < num_elements; ++i)
    {
        out[i] = in1[i] * in2[i];
    }
}

void plast_cpu_mul_kernel_int32(int32_t* out, const int32_t* in1, const int32_t* in2,
                                size_t num_elements)
{
    for (size_t i = 0; i < num_elements; ++i)
    {
        out[i] = in1[i] * in2[i];
    }
}
