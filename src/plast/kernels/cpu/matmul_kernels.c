#include "plast/core/shape_utils_c.h"
#include "plast/kernels/cpu/binary_kernels.h"
#include <immintrin.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define TILE_WIDTH 16

void plast_cpu_matmul_kernel_float(float* out, const float* in1, const float* in2, const int B,
                                   const int N, const int M, const int K)
{
    for (int b = 0; b < B; ++b)
    {
        float* current_out = out + b * N * M;
        const float* current_in1 = in1 + b * N * K;
        const float* current_in2 = in2 + b * K * M;

        // Initialize current_out to zeros
        memset(current_out, 0, N * M * sizeof(float));

        for (int row_tile = 0; row_tile < N; row_tile += 256)
        {
            for (int col_tile = 0; col_tile < M; col_tile += 256)
            {
                for (int inner_tile = 0; inner_tile < K; inner_tile += TILE_WIDTH)
                {
                    for (int row = row_tile; row < fmin(N, row_tile + 256); ++row)
                    {
                        int inner_tile_end = fmin(K, inner_tile + TILE_WIDTH);
                        for (int inner = inner_tile; inner < inner_tile_end; ++inner)
                        {
                            for (int col = col_tile; col < fmin(M, col_tile + 256); ++col)
                            {
                                current_out[row * M + col] +=
                                    current_in1[row * K + inner] * current_in2[inner * M + col];
                            }
                        }
                    }
                }
            }
        }
    }
}

void plast_cpu_matmul_kernel_int32(int32_t* out, const int32_t* in1, const int32_t* in2,
                                   const int B, const int N, const int M, const int K)
{
    for (int b = 0; b < B; ++b)
    {
        int32_t* current_out = out + b * N * M;
        const int32_t* current_in1 = in1 + b * N * K;
        const int32_t* current_in2 = in2 + b * K * M;

        // Initialize current_out to zeros
        memset(current_out, 0, N * M * sizeof(int32_t));

        for (int row_tile = 0; row_tile < N; row_tile += 256)
        {
            for (int col_tile = 0; col_tile < M; col_tile += 256)
            {
                for (int inner_tile = 0; inner_tile < K; inner_tile += TILE_WIDTH)
                {
                    for (int row = row_tile; row < fmin(N, row_tile + 256); ++row)
                    {
                        int inner_tile_end = fmin(K, inner_tile + TILE_WIDTH);
                        for (int inner = inner_tile; inner < inner_tile_end; ++inner)
                        {
                            for (int col = col_tile; col < fmin(M, col_tile + 256); ++col)
                            {
                                current_out[row * M + col] +=
                                    current_in1[row * K + inner] * current_in2[inner * M + col];
                            }
                        }
                    }
                }
            }
        }
    }
}

void plast_cpu_matmul_kernel_strided_float(float* out, const float* in1, const float* in2,
                                           const size_t* out_shape, size_t out_ndim,
                                           const size_t* in1_strides, const size_t* in2_strides,
                                           const size_t* in1_shape, const size_t* in2_shape,
                                           size_t K_dim)
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
        float sum = 0.0f;
        // current_indices now represents the (batch, N, M) indices for the output
        // We need to iterate K times for the dot product

        // Calculate the row and col for the current output element
        size_t out_row = current_indices[out_ndim - 2];
        size_t out_col = current_indices[out_ndim - 1];

        // Calculate the base indices for in1 and in2, excluding the last two dimensions
        size_t in1_base_idx = 0;
        size_t in2_base_idx = 0;
        for (size_t d = 0; d < out_ndim - 2; ++d)
        {
            in1_base_idx += current_indices[d] * in1_strides[d];
            in2_base_idx += current_indices[d] * in2_strides[d];
        }

        for (size_t k = 0; k < K_dim; ++k)
        {
            // Calculate the full index for in1 (..., N, k)
            size_t in1_idx =
                in1_base_idx + out_row * in1_strides[out_ndim - 2] + k * in1_strides[out_ndim - 1];
            // Calculate the full index for in2 (..., k, M)
            size_t in2_idx =
                in2_base_idx + k * in2_strides[out_ndim - 2] + out_col * in2_strides[out_ndim - 1];

            sum += in1[in1_idx] * in2[in2_idx];
        }
        out[i] = sum;

        increment_indices(current_indices, out_shape, out_ndim);
    }
    free(current_indices);
}

void plast_cpu_matmul_kernel_strided_int32(int32_t* out, const int32_t* in1, const int32_t* in2,
                                           const size_t* out_shape, size_t out_ndim,
                                           const size_t* in1_strides, const size_t* in2_strides,
                                           const size_t* in1_shape, const size_t* in2_shape,
                                           size_t K_dim)
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
        int32_t sum = 0;
        // current_indices now represents the (batch, N, M) indices for the output
        // We need to iterate K times for the dot product

        // Calculate the row and col for the current output element
        size_t out_row = current_indices[out_ndim - 2];
        size_t out_col = current_indices[out_ndim - 1];

        // Calculate the base indices for in1 and in2, excluding the last two dimensions
        size_t in1_base_idx = 0;
        size_t in2_base_idx = 0;
        for (size_t d = 0; d < out_ndim - 2; ++d)
        {
            in1_base_idx += current_indices[d] * in1_strides[d];
            in2_base_idx += current_indices[d] * in2_strides[d];
        }

        for (size_t k = 0; k < K_dim; ++k)
        {
            // Calculate the full index for in1 (..., N, k)
            size_t in1_idx =
                in1_base_idx + out_row * in1_strides[out_ndim - 2] + k * in1_strides[out_ndim - 1];
            // Calculate the full index for in2 (..., k, M)
            size_t in2_idx =
                in2_base_idx + k * in2_strides[out_ndim - 2] + out_col * in2_strides[out_ndim - 1];

            sum += in1[in1_idx] * in2[in2_idx];
        }
        out[i] = sum;

        increment_indices(current_indices, out_shape, out_ndim);
    }
    free(current_indices);
}
