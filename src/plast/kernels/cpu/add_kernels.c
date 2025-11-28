#include "plast/kernels/cpu/binary_kernels.h" // For function declarations
#undef __AVX512F__
#undef __AVX512VL__
#include <immintrin.h> // For SIMD instructions
#include <math.h>      // For general math functions
#include <stddef.h>    // For size_t
#include <stdint.h>    // For int32_t
#include <string.h>    // For memcpy (if needed)

// Placeholder for Sleef_powf8_u10 if it's not used in add
// extern __m256 Sleef_powf8_u10(__m256 x, __m256 y);

#define SIMD_WIDTH 8 // For __m256 (float)

// CPU kernel for element-wise addition of float tensors
void plast_cpu_add_kernel_float(float* out, const float* in1, const float* in2, size_t num_elements)
{
    size_t i = 0;
    // Use SIMD for float if available and aligned/contiguous
    // For simplicity, assuming inputs are contiguous and aligned for SIMD
#ifdef __AVX2__
    for (; i + SIMD_WIDTH - 1 < num_elements; i += SIMD_WIDTH)
    {
        __m256 x = _mm256_loadu_ps(in1 + i); // Use loadu_ps for unaligned access
        __m256 y = _mm256_loadu_ps(in2 + i);
        __m256 z = _mm256_add_ps(x, y);
        _mm256_storeu_ps(out + i, z); // Use storeu_ps for unaligned access
    }
#endif
    // Handle remaining elements
    for (; i < num_elements; ++i)
    {
        out[i] = in1[i] + in2[i];
    }
}

// CPU kernel for element-wise addition of int32_t tensors
void plast_cpu_add_kernel_int32(int32_t* out, const int32_t* in1, const int32_t* in2,
                                size_t num_elements)
{
    for (size_t i = 0; i < num_elements; ++i)
    {
        out[i] = in1[i] + in2[i];
    }
}

// Add implementations for other data types as needed
