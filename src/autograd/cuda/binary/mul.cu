#include "autograd/cuda/binary/common.cuh"
#include "autograd/cuda/broadcast_utils.cuh"
#include "device_management.h"

__global__ void mul_grad_kernel(const float* out_grad, float* prev_grad, const float* other_data,
                                int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
    {
        prev_grad[i] += out_grad[i] * other_data[i];
    }
}

__global__ void scalar_mul_grad_kernel(const float* out_grad, float* prev_grad, float scalar, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
    {
        prev_grad[i] += out_grad[i] * scalar;
    }
}

__global__ void noncontig_mul_grad_kernel(const float* out_grad, float* prev_grad,
                                          const float* other_data, int n, const int* prev_shape,
                                          const int* prev_strides, int prev_ndim,
                                          const int* other_shape, const int* other_strides,
                                          int other_ndim, const int* out_shape,
                                          const int* out_strides, int out_ndim)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        int out_offset =
            get_broadcasted_input_idx(i, out_shape, out_ndim, out_shape, out_strides, out_ndim);
        int prev_grad_offset =
            get_broadcasted_input_idx(i, out_shape, out_ndim, prev_shape, prev_strides, prev_ndim);
        int other_data_offset = get_broadcasted_input_idx(i, out_shape, out_ndim, other_shape,
                                                          other_strides, other_ndim);

        atomicAdd(&prev_grad[prev_grad_offset],
                  out_grad[out_offset] * other_data[other_data_offset]);
    }
}

void mul_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("mul_grad_op_cuda: Entering function with n_prev=%d", n_prev);

    assert(out && "Output tensor cannot be NULL");
    assert(out->grad && "Output tensor gradient cannot be NULL");
    assert(prev && "Previous tensors array cannot be NULL");

    int N = numel(out->shape, out->ndim);
    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    if (n_prev == 1) // Tensor * Scalar
    {
        assert(extras && "Extras (scalar value) cannot be NULL for scalar multiplication");
        float scalar = *((float*) extras);
        Tensor* a = prev[0];
        if (a->requires_grad)
        {
            scalar_mul_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                out->grad->data->data, a->grad->data->data, scalar, N);
            CHECK_CUDA();
        }
    }
    else if (n_prev == 2) // Tensor * Tensor
    {
        Tensor* a = prev[0];
        Tensor* b = prev[1];

        // Grad for a: da = dout * b
        if (a->requires_grad)
        {
            if (shapes_equal(a->shape, a->ndim, out->shape, out->ndim) &&
                shapes_equal(b->shape, b->ndim, out->shape, out->ndim) && is_contiguous(a) &&
                is_contiguous(b) && is_contiguous(out))
            {
                mul_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, a->grad->data->data, b->data->data, N);
            }
            else
            {
                int *d_out_shape, *d_out_strides, *d_a_shape, *d_a_strides, *d_b_shape,
                    *d_b_strides;
                copy_shape_and_strides_to_device(out->shape, out->strides, out->ndim, &d_out_shape,
                                                 &d_out_strides);
                copy_shape_and_strides_to_device(a->shape, a->strides, a->ndim, &d_a_shape,
                                                 &d_a_strides);
                copy_shape_and_strides_to_device(b->shape, b->strides, b->ndim, &d_b_shape,
                                                 &d_b_strides);

                noncontig_mul_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, a->grad->data->data, b->data->data, N, d_a_shape,
                    d_a_strides, a->ndim, d_b_shape, d_b_strides, b->ndim, d_out_shape,
                    d_out_strides, out->ndim);

                cudaFree(d_out_shape);
                cudaFree(d_out_strides);
                cudaFree(d_a_shape);
                cudaFree(d_a_strides);
                cudaFree(d_b_shape);
                cudaFree(d_b_strides);
            }
            CHECK_CUDA();
        }

        // Grad for b: db = dout * a
        if (b->requires_grad)
        {
            if (shapes_equal(a->shape, a->ndim, out->shape, out->ndim) &&
                shapes_equal(b->shape, b->ndim, out->shape, out->ndim) && is_contiguous(a) &&
                is_contiguous(b) && is_contiguous(out))
            {
                mul_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, b->grad->data->data, a->data->data, N);
            }
            else
            {
                int *d_out_shape, *d_out_strides, *d_a_shape, *d_a_strides, *d_b_shape,
                    *d_b_strides;
                copy_shape_and_strides_to_device(out->shape, out->strides, out->ndim, &d_out_shape,
                                                 &d_out_strides);
                copy_shape_and_strides_to_device(a->shape, a->strides, a->ndim, &d_a_shape,
                                                 &d_a_strides);
                copy_shape_and_strides_to_device(b->shape, b->strides, b->ndim, &d_b_shape,
                                                 &d_b_strides);

                noncontig_mul_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, b->grad->data->data, a->data->data, N, d_b_shape,
                    d_b_strides, b->ndim, d_a_shape, d_a_strides, a->ndim, d_out_shape,
                    d_out_strides, out->ndim);

                cudaFree(d_out_shape);
                cudaFree(d_out_strides);
                cudaFree(d_a_shape);
                cudaFree(d_a_strides);
                cudaFree(d_b_shape);
                cudaFree(d_b_strides);
            }
            CHECK_CUDA();
        }
    }
}
