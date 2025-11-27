#include "plast/kernels/cpu/binary_kernels.h" // For function declarations
#include <immintrin.h> // For SIMD instructions
#include <math.h>      // For general math functions
#include <string.h>    // For memcpy (if needed)
#include <stddef.h>    // For size_t
#include <stdint.h>    // For int32_t

#define SIMD_WIDTH 8 // For __m256 (float)

// CPU kernel for element-wise subtraction of float tensors
void plast_cpu_sub_kernel_float(float* out, const float* in1, const float* in2, size_t num_elements) {
    size_t i = 0;
    // Use SIMD for float if available and aligned/contiguous
    for (; i + SIMD_WIDTH - 1 < num_elements; i += SIMD_WIDTH) {
        __m256 x = _mm256_loadu_ps(in1 + i); // Use loadu_ps for unaligned access
        __m256 y = _mm256_loadu_ps(in2 + i);
        __m256 z = _mm256_sub_ps(x, y);
        _mm256_storeu_ps(out + i, z); // Use storeu_ps for unaligned access
    }
    // Handle remaining elements
    for (; i < num_elements; ++i) {
        out[i] = in1[i] - in2[i];
    }
}

// CPU kernel for element-wise subtraction of int32_t tensors
void plast_cpu_sub_kernel_int32(int32_t* out, const int32_t* in1, const int32_t* in2, size_t num_elements) {
    // No SIMD for int32_t for now, simple loop
    for (size_t i = 0; i < num_elements; ++i) {
        out[i] = in1[i] - in2[i];
    }
}
