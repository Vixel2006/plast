#include "autograd/autograd_utils.h"
#include "autograd/cuda/broadcast_utils.cuh"
#include "autograd/cuda/unary/common.cuh"
#include "autograd/cuda/unary/unary_ops_cuda.h"
#include "device_management.h"

__global__ void clip_grad_kernel(const float* out_grad, const float* prev_data, float* prev_grad,
                                 float min_val, float max_val, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        float x = prev_data[i];
        float mask = (x >= min_val) && (x <= max_val);
        prev_grad[i] += out_grad[i] * mask;
    }
}

__global__ void noncontig_clip_grad_kernel(const float* out_grad, const float* prev_data,
                                           float* prev_grad, float min_val, float max_val, int n,
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
        float x = prev_data[prev_offset];
        float mask = (x >= min_val) && (x <= max_val);
        atomicAdd(&prev_grad[prev_offset], out_grad[out_offset] * mask);
    }
}

void clip_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("clip_grad_op_cuda: Entering function with n_prev=%d", n_prev);

    assert(out && out->grad && out->grad->data && out->grad->data->data);
    assert(prev && n_prev == 1 && prev[0] && prev[0]->data && prev[0]->data->data);
    assert(extras);

    Tensor* a = prev[0];
    if (!a->requires_grad)
    {
        return;
    }
    assert(a->grad && a->grad->data && a->grad->data->data);

    ClipExtras* clip_extras = (ClipExtras*) extras;
    float min_val = clip_extras->min_val;
    float max_val = clip_extras->max_val;

    int N = numel(out->shape, out->ndim);
    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    if (is_contiguous(a) && is_contiguous(out))
    {
        clip_grad_kernel<<<num_blocks, num_threads_per_block>>>(
            out->grad->data->data, a->data->data, a->grad->data->data, min_val, max_val, N);
    }
    else
    {
        int *d_out_shape, *d_out_strides, *d_prev_shape, *d_prev_strides;
        copy_shape_and_strides_to_device(out->shape, out->strides, out->ndim, &d_out_shape,
                                         &d_out_strides);
        copy_shape_and_strides_to_device(a->shape, a->strides, a->ndim, &d_prev_shape,
                                         &d_prev_strides);

        noncontig_clip_grad_kernel<<<num_blocks, num_threads_per_block>>>(
            out->grad->data->data, a->data->data, a->grad->data->data, min_val, max_val, N,
            d_prev_shape, d_prev_strides, a->ndim, d_out_shape, d_out_strides, out->ndim);

        cudaFree(d_out_shape);
        cudaFree(d_out_strides);
        cudaFree(d_prev_shape);
        cudaFree(d_prev_strides);
    }
    CHECK_CUDA();
}
