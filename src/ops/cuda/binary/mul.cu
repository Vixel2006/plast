#include "ops/cuda/binary.h"
#include "ops/cuda/init.h"

__device__ inline int get_bcast_offset(int linear_idx, const int* out_shape, int out_ndim,
                                    const int* in_strides, int in_ndim)
{
    int offset = 0;
    for (int i = out_ndim - 1; i >= 0; --i)
    {
        int coord = linear_idx % out_shape[i];
        linear_idx /= out_shape[i];

        int in_dim_idx = i - (out_ndim - in_ndim);
        if (in_dim_idx >= 0)
        {
            offset += coord * in_strides[in_dim_idx];
        }
    }
    return offset;
}

__global__ void broadcasted_mul_kernel(const float* a, const float* b, float* out, const int n,
                                     const int* a_strides, int a_ndim, const int* b_strides,
                                     int b_ndim, const int* out_shape, const int* out_strides,
                                     int out_ndim)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        int a_offset = get_bcast_offset(i, out_shape, out_ndim, a_strides, a_ndim);
        int b_offset = get_bcast_offset(i, out_shape, out_ndim, b_strides, b_ndim);
        int out_offset = get_bcast_offset(i, out_shape, out_ndim, out_strides, out_ndim);

        out[out_offset] = a[a_offset] * b[b_offset];
    }
}

__global__ void contig_mul_kernel(const float* a, const float* b, float* out, const int n)
{
    int idx = blockIdx.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        out[i] = a[i] * b[i];
    }
}

extern "C" void mul_op_cuda(Tensor* a, Tensor* b, Tensor* out)
{
    LOG_INFO("mul_op_cuda: Entering function");
    int N = numel(out->shape, out->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    out->data = (Storage*) malloc(sizeof(Storage));
    if (!out->data)
    {
        LOG_ERROR("Failed to allocate Storage for out tensor in mul_op_cuda");
        assert(0 && "Failed to allocate Storage for out tensor in mul_op_cuda");
    }
    out->data->counter = 1;
    out->data->size = N;
    cudaError_t err = cudaMalloc((void**) &out->data->data, out->data->size * sizeof(float));
    if (err != cudaSuccess)
    {
        LOG_ERROR("Failed to allocate CUDA memory for out->data->data in mul_op_cuda: %s",
                  cudaGetErrorString(err));
        SAFE_FREE(&out->data, free);
        assert(0 && "Failed to allocate CUDA memory for out->data->data in mul_op_cuda");
    }

    if (is_contiguous(a) && is_contiguous(b) && is_contiguous(out))
    {
        contig_mul_kernel<<<num_blocks, num_threads_per_block>>>(a->data->data, b->data->data,
                                                                 out->data->data, N);
    }
    else
    {
        int *d_a_strides, *d_b_strides, *d_out_shape, *d_out_strides;
        cudaMalloc(&d_a_strides, a->ndim * sizeof(int));
        cudaMemcpy(d_a_strides, a->strides, a->ndim * sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc(&d_b_strides, b->ndim * sizeof(int));
        cudaMemcpy(d_b_strides, b->strides, b->ndim * sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc(&d_out_shape, out->ndim * sizeof(int));
        cudaMemcpy(d_out_shape, out->shape, out->ndim * sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc(&d_out_strides, out->ndim * sizeof(int));
        cudaMemcpy(d_out_strides, out->strides, out->ndim * sizeof(int), cudaMemcpyHostToDevice);

        broadcasted_mul_kernel<<<num_blocks, num_threads_per_block>>>(
            a->data->data, b->data->data, out->data->data, N, d_a_strides, a->ndim, d_b_strides,
            b->ndim, d_out_shape, d_out_strides, out->ndim);

        cudaFree(d_a_strides);
        cudaFree(d_b_strides);
        cudaFree(d_out_shape);
        cudaFree(d_out_strides);
    }
    CHECK_CUDA();

    LOG_INFO("Mul kernel done successfully");
}
