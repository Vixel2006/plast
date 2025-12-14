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

