#include "autograd/cuda/binary/common.cuh"
#include "autograd/cuda/broadcast_utils.cuh"
#include "device_management.h"

__global__ void scalar_pow_grad_kernel(const float* out_grad, const float* prev_data,
                                       float* prev_grad, float power, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        prev_grad[i] += power * powf(prev_data[i] + 1e-7f, power - 1) * out_grad[i];
    }
}

__global__ void noncontig_scalar_pow_grad_kernel(const float* out_grad, const float* prev_data,
                                                 float* prev_grad, float power, int n,
                                                 const int* prev_shape, const int* prev_strides,
                                                 int prev_ndim, const int* out_shape,
                                                 const int* out_strides, int out_ndim)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        int out_offset =
            get_broadcasted_input_idx(i, out_shape, out_ndim, out_shape, out_strides, out_ndim);
        int prev_offset =
            get_broadcasted_input_idx(i, out_shape, out_ndim, prev_shape, prev_strides, prev_ndim);
        atomicAdd(&prev_grad[prev_offset],
                  power * powf(prev_data[prev_offset] + 1e-7f, power - 1) * out_grad[out_offset]);
    }
}

__global__ void base_pow_grad_kernel(const float* out_grad, const float* base_data,
                                     float* base_grad, const float* power_data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        base_grad[i] += power_data[i] * powf(base_data[i] + 1e-7f, power_data[i] - 1) * out_grad[i];
    }
}

__global__ void noncontig_base_pow_grad_kernel(const float* out_grad, const float* base_data,
                                               float* base_grad, const float* power_data, int n,
                                               const int* base_shape, const int* base_strides,
                                               int base_ndim, const int* power_shape,
                                               const int* power_strides, int power_ndim,
                                               const int* out_shape, const int* out_strides,
                                               int out_ndim)
{
    int idx = blockIdx.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        int out_offset =
            get_broadcasted_input_idx(i, out_shape, out_ndim, out_shape, out_strides, out_ndim);
        int base_offset =
            get_broadcasted_input_idx(i, out_shape, out_ndim, base_shape, base_strides, base_ndim);
        int power_offset = get_broadcasted_input_idx(i, out_shape, out_ndim, power_shape,
                                                     power_strides, power_ndim);
        atomicAdd(&base_grad[base_offset],
                  power_data[power_offset] *
                      powf(base_data[base_offset] + 1e-7f, power_data[power_offset] - 1) *
                      out_grad[out_offset]);
    }
}

__global__ void exponent_pow_grad_kernel(const float* out_grad, const float* out_data,
                                         const float* base_data, float* power_grad, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        power_grad[i] += out_grad[i] * out_data[i] * logf(base_data[i] + 1e-7f);
    }
}

__global__ void noncontig_exponent_pow_grad_kernel(const float* out_grad, const float* out_data,
                                                   const float* base_data, float* power_grad, int n,
                                                   const int* base_shape, const int* base_strides,
                                                   int base_ndim, const int* power_shape,
                                                   const int* power_strides, int power_ndim,
                                                   const int* out_shape, const int* out_strides,
                                                   int out_ndim)
{
    int idx = blockIdx.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        int out_offset =
            get_broadcasted_input_idx(i, out_shape, out_ndim, out_shape, out_strides, out_ndim);
        int base_offset =
            get_broadcasted_input_idx(i, out_shape, out_ndim, base_shape, base_strides, base_ndim);
        int power_offset = get_broadcasted_input_idx(i, out_shape, out_ndim, power_shape,
                                                     power_strides, power_ndim);
        atomicAdd(&power_grad[power_offset], out_grad[out_offset] * out_data[out_offset] *
                                                 logf(base_data[base_offset] + 1e-7f));
    }
}

void pow_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("pow_grad_op_cuda: Entering function with n_prev=%d", n_prev);

    assert(out && "Output tensor cannot be NULL");
    assert(out->grad && "Output tensor gradient cannot be NULL");
    assert(prev && "Previous tensors array cannot be NULL");

    int N = numel(out->shape, out->ndim);
    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    if (n_prev == 1) // prev[0] ** scalar
    {
        assert(extras && "Extras (scalar value) cannot be NULL for scalar power");
        float scalar_power = *((float*) extras);
        Tensor* a = prev[0];
        if (a->requires_grad)
        {
            if (shapes_equal(a->shape, a->ndim, out->shape, out->ndim) && is_contiguous(a) &&
                is_contiguous(out))
            {
                scalar_pow_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, a->data->data, a->grad->data->data, scalar_power, N);
            }
            else
            {
                int *d_out_shape, *d_out_strides, *d_prev_shape, *d_prev_strides;
                copy_shape_and_strides_to_device(out->shape, out->strides, out->ndim, &d_out_shape,
                                                 &d_out_strides);
                copy_shape_and_strides_to_device(a->shape, a->strides, a->ndim, &d_prev_shape,
                                                 &d_prev_strides);

                noncontig_scalar_pow_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, a->data->data, a->grad->data->data, scalar_power, N,
                    d_prev_shape, d_prev_strides, a->ndim, d_out_shape, d_out_strides, out->ndim);

                cudaFree(d_out_shape);
                cudaFree(d_out_strides);
                cudaFree(d_prev_shape);
                cudaFree(d_prev_strides);
            }
            CHECK_CUDA();
        }
    }
    else // base ** power
    {
        Tensor* a = prev[0];
        Tensor* b = prev[1];

        bool all_contiguous_and_equal = shapes_equal(a->shape, a->ndim, out->shape, out->ndim) &&
                                        shapes_equal(b->shape, b->ndim, out->shape, out->ndim) &&
                                        is_contiguous(a) && is_contiguous(b) && is_contiguous(out);

        if (a->requires_grad) // gradient for base
        {
            if (all_contiguous_and_equal)
            {
                base_pow_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, a->data->data, a->grad->data->data, b->data->data, N);
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

                noncontig_base_pow_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, a->data->data, a->grad->data->data, b->data->data, N,
                    d_a_shape, d_a_strides, a->ndim, d_b_shape, d_b_strides, b->ndim, d_out_shape,
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

        if (b->requires_grad) // gradient for power
        {
            if (all_contiguous_and_equal)
            {
                exponent_pow_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, out->data->data, a->data->data, b->grad->data->data, N);
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

                noncontig_exponent_pow_grad_kernel<<<num_blocks, num_threads_per_block>>>(
                    out->grad->data->data, out->data->data, a->data->data, b->grad->data->data, N,
                    d_a_shape, d_a_strides, a->ndim, d_b_shape, d_b_strides, b->ndim, d_out_shape,
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
