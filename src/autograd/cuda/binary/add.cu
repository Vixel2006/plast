#include "autograd/cuda/binary/common.cuh"
#include "autograd/cuda/broadcast_utils.cuh"

__global__ void contig_add_grad_kernel(const float* out_grad, float* prev_grad, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        prev_grad[i] += out_grad[i];
    }
}

__global__ void noncontig_add_grad_kernel(const float* out_grad, float* prev_grad, int n,
                                          const int* prev_shape, const int* prev_strides,
                                          int prev_ndim, const int* out_shape,
                                          const int* out_strides, int out_ndim)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        int in_idx =
            get_broadcasted_input_idx(i, out_shape, out_ndim, prev_shape, prev_strides, prev_ndim);
        atomicAdd(&prev_grad[in_idx], out_grad[i]);
    }
}

void add_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("add_grad_op_cuda: Entering function with n_prev=%d", n_prev);

    assert(out && "Output tensor cannot be NULL");
    assert(out->grad && "Output tensor gradient cannot be NULL");
    assert(out->grad->data && "Output tensor gradient data cannot be NULL");
    assert(out->grad->data->data && "Output tensor gradient data pointer cannot be NULL");
    assert(prev && "Previous tensors array cannot be NULL");
    assert((n_prev == 1 || n_prev == 2) && "n_prev must be 1 or 2 for add_grad_op_cuda");

    int N = numel(out->shape, out->ndim);
    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    for (int i = 0; i < n_prev; ++i)
    {
        Tensor* p = prev[i];
        if (p->requires_grad)
        {
            if (shapes_equal(p->shape, p->ndim, out->shape, out->ndim) && is_contiguous(p) &&
                is_contiguous(out))
            {
                contig_add_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, p->grad->data->data, N);
            }
            else
            {
                int *d_out_shape, *d_out_strides, *d_prev_shape, *d_prev_strides;

                cudaMalloc(&d_out_shape, out->ndim * sizeof(int));
                cudaMemcpy(d_out_shape, out->shape, out->ndim * sizeof(int), cudaMemcpyHostToDevice);

                cudaMalloc(&d_out_strides, out->ndim * sizeof(int));
                cudaMemcpy(d_out_strides, out->strides, out->ndim * sizeof(int),
                           cudaMemcpyHostToDevice);

                cudaMalloc(&d_prev_shape, p->ndim * sizeof(int));
                cudaMemcpy(d_prev_shape, p->shape, p->ndim * sizeof(int), cudaMemcpyHostToDevice);

                cudaMalloc(&d_prev_strides, p->ndim * sizeof(int));
                cudaMemcpy(d_prev_strides, p->strides, p->ndim * sizeof(int),
                           cudaMemcpyHostToDevice);

                noncontig_add_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, p->grad->data->data, N, d_prev_shape, d_prev_strides,
                    p->ndim, d_out_shape, d_out_strides, out->ndim);

                cudaFree(d_out_shape);
                cudaFree(d_out_strides);
                cudaFree(d_prev_shape);
                cudaFree(d_prev_strides);
            }
            CHECK_CUDA();
        }
    }
}
