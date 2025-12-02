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

void plast_cpu_sub_kernel_strided_float(float* out, const float* in1, const float* in2,
                                        const size_t* out_shape, size_t out_ndim,
                                        const size_t* in1_strides, const size_t* in2_strides)
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
        size_t in1_idx = get_index(current_indices, in1_strides, out_ndim);
        size_t in2_idx = get_index(current_indices, in2_strides, out_ndim);
        out[i] = in1[in1_idx] - in2[in2_idx];

        // Increment indices
        increment_indices(current_indices, out_shape, out_ndim);
    }
    free(current_indices);
}

void plast_cpu_sub_kernel_strided_int32(int32_t* out, const int32_t* in1, const int32_t* in2,
                                        const size_t* out_shape, size_t out_ndim,
                                        const size_t* in1_strides, const size_t* in2_strides)
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
        size_t in1_idx = get_index(current_indices, in1_strides, out_ndim);
        size_t in2_idx = get_index(current_indices, in2_strides, out_ndim);
        out[i] = in1[in1_idx] - in2[in2_idx];

        // Increment indices
        increment_indices(current_indices, out_shape, out_ndim);
    }
    free(current_indices);
}
