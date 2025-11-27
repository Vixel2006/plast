#include "ops/cuda/binary.h"
#include "ops/cuda/init.h"

__global__ void broadcasted_matmul_kernel(const float* a, const float* b, float* out, const int N,
                                          const int K, const int P, const int* a_strides,
                                          int a_ndim, const int* b_strides, int b_ndim,
                                          const int* out_shape, const int* out_strides, int out_ndim)
{
    int batch = blockIdx.z;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    // Calculate batch offsets
    int a_batch_offset = 0;
    int b_batch_offset = 0;
    int out_batch_offset = 0;

    int temp_batch_idx = batch;
    for (int dim = out_ndim - 3; dim >= 0; --dim)
    {
        int coord = temp_batch_idx % out_shape[dim];
        temp_batch_idx /= out_shape[dim];

        int a_dim_idx = dim - (out_ndim - a_ndim);
        if (a_dim_idx >= 0)
        {
            a_batch_offset += coord * a_strides[a_dim_idx];
        }

        int b_dim_idx = dim - (out_ndim - b_ndim);
        if (b_dim_idx >= 0)
        {
            b_batch_offset += coord * b_strides[b_dim_idx];
        }
        out_batch_offset += coord * out_strides[dim];
    }

    __shared__ float a_tile[TILE_DIM][TILE_DIM];
    __shared__ float b_tile[TILE_DIM][TILE_DIM];

    float sum = 0.0f;

    const float* a_batch = a + a_batch_offset;
    const float* b_batch = b + b_batch_offset;
    float* c_batch = out + out_batch_offset;

    int a_row_stride = a_strides[a_ndim - 2];
    int a_col_stride = a_strides[a_ndim - 1];
    int b_row_stride = b_strides[b_ndim - 2];
    int b_col_stride = b_strides[b_ndim - 1];

    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t)
    {
        int tiledColA = t * TILE_DIM + threadIdx.x;
        int tiledRowB = t * TILE_DIM + threadIdx.y;

        if (row < N && tiledColA < K)
        {
            a_tile[threadIdx.y][threadIdx.x] = a_batch[row * a_row_stride + tiledColA * a_col_stride];
        }
        else
        {
            a_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (tiledRowB < K && col < P)
        {
            b_tile[threadIdx.y][threadIdx.x] =
                b_batch[tiledRowB * b_row_stride + col * b_col_stride];
        }
        else
        {
            b_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k)
        {
            sum += a_tile[threadIdx.y][k] * b_tile[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (col < P && row < N)
    {
        c_batch[row * out_strides[out_ndim - 2] + col * out_strides[out_ndim - 1]] = sum;
    }
}

extern "C" void matmul_op_cuda(Tensor* a, Tensor* b, Tensor* out, int N, int K, int P)
{
    LOG_INFO("matmul_op_cuda: Entering function with N=%d, K=%d, P=%d", N, K, P);

    int B = 1;
    if (out->ndim > 2)
    {
        for (int i = 0; i < out->ndim - 2; ++i)
        {
            B *= out->shape[i];
        }
    }

    out->data = (Storage*) malloc(sizeof(Storage));
    if (!out->data)
    {
        LOG_ERROR("Failed to allocate Storage for out tensor in matmul_op_cuda");
        assert(0 && "Failed to allocate Storage for out tensor in matmul_op_cuda");
    }
    out->data->counter = 1;
    out->data->size = B * N * P;
    cudaError_t err = cudaMalloc((void**) &out->data->data, out->data->size * sizeof(float));
    if (err != cudaSuccess)
    {
        LOG_ERROR("Failed to allocate CUDA memory for out->data->data in matmul_op_cuda: %s",
                  cudaGetErrorString(err));
        SAFE_FREE(&out->data, free);
        assert(0 && "Failed to allocate CUDA memory for out->data->data in matmul_op_cuda");
    }

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((P + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM, B);

    int *d_a_strides, *d_b_strides, *d_out_shape, *d_out_strides;
    cudaMalloc(&d_a_strides, a->ndim * sizeof(int));
    cudaMemcpy(d_a_strides, a->strides, a->ndim * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_b_strides, b->ndim * sizeof(int));
    cudaMemcpy(d_b_strides, b->strides, b->ndim * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_out_shape, out->ndim * sizeof(int));
    cudaMemcpy(d_out_shape, out->shape, out->ndim * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_out_strides, out->ndim * sizeof(int));
    cudaMemcpy(d_out_strides, out->strides, out->ndim * sizeof(int), cudaMemcpyHostToDevice);

    broadcasted_matmul_kernel<<<grid, block>>>(
        a->data->data, b->data->data, out->data->data, N, K, P, d_a_strides, a->ndim, d_b_strides,
        b->ndim, d_out_shape, d_out_strides, out->ndim);

    cudaFree(d_a_strides);
    cudaFree(d_b_strides);
    cudaFree(d_out_shape);
    cudaFree(d_out_strides);

    CHECK_CUDA();

    LOG_INFO("MATMUL kernel done successfully");
}
