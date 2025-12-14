#include "plast/core/types.h"
#include "plast/kernels/cuda/binary_kernels.h"
#include <cuda_runtime.h>
#include <stdexcept>

#define TILE_DIM 16

__global__ void matmul_cuda_kernel_float(float* out, const float* in1, const float* in2, int B_dim,
                                         int N_dim, int M_dim, int K_dim)
{
    __shared__ float s_in1[TILE_DIM][TILE_DIM];
    __shared__ float s_in2[TILE_DIM][TILE_DIM];

    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int batch = blockIdx.z;

    float sum = 0.0f;

    for (int tile_k_idx = 0; tile_k_idx < (K_dim + TILE_DIM - 1) / TILE_DIM; ++tile_k_idx)
    {
        int in1_global_row = blockIdx.y * TILE_DIM + threadIdx.y;
        int in1_global_col = tile_k_idx * TILE_DIM + threadIdx.x;

        if (in1_global_row < N_dim && in1_global_col < K_dim)
        {
            s_in1[threadIdx.y][threadIdx.x] =
                in1[batch * N_dim * K_dim + in1_global_row * K_dim + in1_global_col];
        }
        else
        {
            s_in1[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int in2_global_row = tile_k_idx * TILE_DIM + threadIdx.y;
        int in2_global_col = blockIdx.x * TILE_DIM + threadIdx.x;

        if (in2_global_row < K_dim && in2_global_col < M_dim)
        {
            s_in2[threadIdx.x][threadIdx.y] =
                in2[batch * K_dim * M_dim + in2_global_row * M_dim + in2_global_col];
        }
        else
        {
            s_in2[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k)
        {
            sum += s_in1[threadIdx.y][k] * s_in2[threadIdx.x][k];
        }
        __syncthreads();
    }

    if (row < N_dim && col < M_dim)
    {
        out[batch * N_dim * M_dim + row * M_dim + col] = sum;
    }
}

extern "C" void plast_cuda_matmul_kernel_float(float* out, const float* in1, const float* in2,
                                               int B, int N, int M, int K)
{
    // Define block and grid dimensions for a batched tiled matrix multiplication.
    // Each block processes a TILE_DIM x TILE_DIM sub-matrix of the output for a single batch.
    // grid.x: number of blocks needed along the M dimension (columns of output)
    // grid.y: number of blocks needed along the N dimension (rows of output)
    // grid.z: number of batches
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((M + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM, B);

    // Launch the CUDA kernel.
    matmul_cuda_kernel_float<<<grid, block>>>(out, in1, in2, B, N, M, K);

    // Check for any CUDA errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("CUDA error in plast_cuda_matmul_kernel_float: ") +
                                 cudaGetErrorString(err));
    }
}

__global__ void matmul_cuda_kernel_int32(int32_t* out, const int32_t* in1, const int32_t* in2,
                                         int B_dim, int N_dim, int M_dim, int K_dim)
{
    __shared__ int32_t s_in1[TILE_DIM][TILE_DIM];
    __shared__ int32_t s_in2[TILE_DIM][TILE_DIM];

    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int batch = blockIdx.z;

    int32_t sum = 0;

    for (int tile_k_idx = 0; tile_k_idx < (K_dim + TILE_DIM - 1) / TILE_DIM; ++tile_k_idx)
    {
        int in1_global_row = blockIdx.y * TILE_DIM + threadIdx.y;
        int in1_global_col = tile_k_idx * TILE_DIM + threadIdx.x;

        if (in1_global_row < N_dim && in1_global_col < K_dim)
        {
            s_in1[threadIdx.y][threadIdx.x] =
                in1[batch * N_dim * K_dim + in1_global_row * K_dim + in1_global_col];
        }
        else
        {
            s_in1[threadIdx.y][threadIdx.x] = 0;
        }

        int in2_global_row = tile_k_idx * TILE_DIM + threadIdx.y;
        int in2_global_col = blockIdx.x * TILE_DIM + threadIdx.x;

        if (in2_global_row < K_dim && in2_global_col < M_dim)
        {
            s_in2[threadIdx.x][threadIdx.y] = // Store transposed
                in2[batch * K_dim * M_dim + in2_global_row * M_dim + in2_global_col];
        }
        else
        {
            s_in2[threadIdx.x][threadIdx.y] = 0; // Store transposed
        }

        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k)
        {
            sum += s_in1[threadIdx.y][k] * s_in2[threadIdx.x][k]; // Access transposed s_in2
        }
        __syncthreads();
    }

    if (row < N_dim && col < M_dim)
    {
        out[batch * N_dim * M_dim + row * M_dim + col] = sum;
    }
}

extern "C" void plast_cuda_matmul_kernel_int32(int32_t* out, const int32_t* in1, const int32_t* in2,
                                               int B, int N, int M, int K)
{
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((M + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM, B);

    matmul_cuda_kernel_int32<<<grid, block>>>(out, in1, in2, B, N, M, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("CUDA error in plast_cuda_matmul_kernel_int32: ") +
                                 cudaGetErrorString(err));
    }
}

