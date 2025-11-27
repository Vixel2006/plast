#include "autograd/cuda/binary/common.cuh"
#include "device_management.h"
#include "tensor.h"

__global__ void broadcasted_matmul_grad_kernel(
    const float* lhs, const float* rhs, float* grad, int N, int P, int K, bool transpose_lhs,
    bool transpose_rhs, const int* lhs_shape, const int* lhs_strides, int lhs_ndim,
    const int* rhs_shape, const int* rhs_strides, int rhs_ndim, const int* grad_shape,
    const int* grad_strides, int grad_ndim, const int* out_shape, int out_ndim)
{
    __shared__ float lhs_tile[TILE_DIM][TILE_DIM];
    __shared__ float rhs_tile[TILE_DIM][TILE_DIM];

    int batch = blockIdx.z;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    float sum = 0.0f;

    // --- Batch Offset Calculation ---
    int lhs_batch_offset = 0;
    int rhs_batch_offset = 0;
    int grad_batch_offset = 0;
    int temp_batch_idx = batch;

    for (int dim = out_ndim - 3; dim >= 0; --dim)
    {
        int coord = temp_batch_idx % out_shape[dim];
        temp_batch_idx /= out_shape[dim];

        if (dim - (out_ndim - lhs_ndim) >= 0)
            lhs_batch_offset += coord * lhs_strides[dim - (out_ndim - lhs_ndim)];
        if (dim - (out_ndim - rhs_ndim) >= 0)
            rhs_batch_offset += coord * rhs_strides[dim - (out_ndim - rhs_ndim)];
        if (dim - (out_ndim - grad_ndim) >= 0)
            grad_batch_offset += coord * grad_strides[dim - (out_ndim - grad_ndim)];
    }

    const float* batched_lhs = lhs + lhs_batch_offset;
    const float* batched_rhs = rhs + rhs_batch_offset;
    float* grad_ptr = grad + grad_batch_offset;

    // --- Strides for Matrix Dimensions ---
    int lhs_row_stride = transpose_lhs ? lhs_strides[lhs_ndim - 1] : lhs_strides[lhs_ndim - 2];
    int lhs_col_stride = transpose_lhs ? lhs_strides[lhs_ndim - 2] : lhs_strides[lhs_ndim - 1];
    int rhs_row_stride = transpose_rhs ? rhs_strides[rhs_ndim - 1] : rhs_strides[rhs_ndim - 2];
    int rhs_col_stride = transpose_rhs ? rhs_strides[rhs_ndim - 2] : rhs_strides[rhs_ndim - 1];

    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t)
    {
        int k_lhs = t * TILE_DIM + threadIdx.x;
        int k_rhs = t * TILE_DIM + threadIdx.y;

        if (row < N && k_lhs < K)
            lhs_tile[threadIdx.y][threadIdx.x] =
                batched_lhs[row * lhs_row_stride + k_lhs * lhs_col_stride];
        else
            lhs_tile[threadIdx.y][threadIdx.x] = 0.0f;

        if (k_rhs < K && col < P)
            rhs_tile[threadIdx.y][threadIdx.x] =
                batched_rhs[k_rhs * rhs_row_stride + col * rhs_col_stride];
        else
            rhs_tile[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k_tile = 0; k_tile < TILE_DIM; ++k_tile)
        {
            sum += lhs_tile[threadIdx.y][k_tile] * rhs_tile[k_tile][threadIdx.x];
        }
        __syncthreads();
    }

    if (col < P && row < N)
    {
        int grad_offset = row * grad_strides[grad_ndim - 2] + col * grad_strides[grad_ndim - 1];
        atomicAdd(&grad_ptr[grad_offset], sum);
    }
}

void matmul_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("matmul_grad_op_cuda: Entering function with n_prev=%d", n_prev);

    assert(out && out->grad && prev && n_prev == 2);
    MatMulBackwardExtras* matmul_extras = (MatMulBackwardExtras*) extras;
    assert(matmul_extras);

    Tensor* A = prev[0];
    Tensor* B = prev[1];
    assert(A && B && A->data && B->data && out->grad->data);

    int N = matmul_extras->N;
    int K = matmul_extras->K;
    int M = matmul_extras->M;

    int B_dim = 1;
    for (int i = 0; i < out->ndim - 2; ++i)
    {
        B_dim *= out->shape[i];
    }

    // Allocate device memory for shapes and strides
    int *d_A_shape, *d_A_strides, *d_B_shape, *d_B_strides, *d_out_grad_shape, *d_out_grad_strides,
        *d_A_grad_shape, *d_A_grad_strides, *d_B_grad_shape, *d_B_grad_strides, *d_out_shape;

    copy_shape_and_strides_to_device(A->shape, A->strides, A->ndim, &d_A_shape, &d_A_strides);
    copy_shape_and_strides_to_device(B->shape, B->strides, B->ndim, &d_B_shape, &d_B_strides);
    copy_shape_and_strides_to_device(out->grad->shape, out->grad->strides, out->grad->ndim,
                                     &d_out_grad_shape, &d_out_grad_strides);
    cudaMalloc(&d_out_shape, out->ndim * sizeof(int));
    cudaMemcpy(d_out_shape, out->shape, out->ndim * sizeof(int), cudaMemcpyHostToDevice);

    if (A->requires_grad)
    {
        assert(A->grad && A->grad->data);
        copy_shape_and_strides_to_device(A->grad->shape, A->grad->strides, A->grad->ndim,
                                         &d_A_grad_shape, &d_A_grad_strides);

        dim3 grid((K + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM, B_dim);
        dim3 block(TILE_DIM, TILE_DIM);

        // dA = dC @ B.T
        broadcasted_matmul_grad_kernel<<<grid, block>>>(
            out->grad->data->data, B->data->data, A->grad->data->data, N, K, M, false, true,
            d_out_grad_shape, d_out_grad_strides, out->grad->ndim, d_B_shape, d_B_strides, B->ndim,
            d_A_grad_shape, d_A_grad_strides, A->grad->ndim, d_out_shape, out->ndim);
        CHECK_CUDA();
        cudaFree(d_A_grad_shape);
        cudaFree(d_A_grad_strides);
    }

    if (B->requires_grad)
    {
        assert(B->grad && B->grad->data);
        copy_shape_and_strides_to_device(B->grad->shape, B->grad->strides, B->grad->ndim,
                                         &d_B_grad_shape, &d_B_grad_strides);

        dim3 grid((M + TILE_DIM - 1) / TILE_DIM, (K + TILE_DIM - 1) / TILE_DIM, B_dim);
        dim3 block(TILE_DIM, TILE_DIM);

        // dB = A.T @ dC
        broadcasted_matmul_grad_kernel<<<grid, block>>>(
            A->data->data, out->grad->data->data, B->grad->data->data, K, M, N, true, false,
            d_A_shape, d_A_strides, A->ndim, d_out_grad_shape, d_out_grad_strides, out->grad->ndim,
            d_B_grad_shape, d_B_grad_strides, B->grad->ndim, d_out_shape, out->ndim);
        CHECK_CUDA();
        cudaFree(d_B_grad_shape);
        cudaFree(d_B_grad_strides);
    }

    cudaFree(d_A_shape);
    cudaFree(d_A_strides);
    cudaFree(d_B_shape);
    cudaFree(d_B_strides);
    cudaFree(d_out_grad_shape);
    cudaFree(d_out_grad_strides);
    cudaFree(d_out_shape);
}
