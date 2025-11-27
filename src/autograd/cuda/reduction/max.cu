#include "autograd/cuda/reduction/common.cuh"
#include "autograd/cuda/reduction/reduction_ops_cuda.h"
#include "core_types.h"
#include "device_management.h"

__global__ void max_grad_kernel(const float* out_grad, float* in_grad, const float* in_data,
                                const float* out_data, int n_in, const int* in_shape,
                                const int* in_strides, int in_ndim, const int* out_strides,
                                int out_ndim, int reduced_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_in) return;

    bool keepdim = (in_ndim == out_ndim);

    // Get input coords from linear index
    int in_coords[MAX_NDIM];
    int temp_idx = idx;
    for (int d = in_ndim - 1; d >= 0; --d)
    {
        in_coords[d] = temp_idx % in_shape[d];
        temp_idx /= in_shape[d];
    }

    // Get in_grad and in_data offset
    int in_offset = 0;
    for (int d = 0; d < in_ndim; ++d)
    {
        in_offset += in_coords[d] * in_strides[d];
    }

    // Get out_grad and out_data offset
    int out_offset = 0;
    int out_d = 0;
    for (int d = 0; d < in_ndim; ++d)
    {
        if (d == reduced_dim && !keepdim)
        {
            continue;
        }
        int coord = (d == reduced_dim && keepdim) ? 0 : in_coords[d];
        out_offset += coord * out_strides[out_d];
        out_d++;
    }

    if (fabsf(in_data[in_offset] - out_data[out_offset]) < EPSILON)
    {
        in_grad[in_offset] += out_grad[out_offset];
    }
}

void max_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("max_grad_op_cuda: Entering function with n_prev=%d", n_prev);

    assert(out && out->grad && out->grad->data && out->grad->data->data);
    assert(out->data && out->data->data);
    assert(prev && n_prev == 1 && prev[0]);
    assert(extras);

    Tensor* a = prev[0];
    if (!a->requires_grad)
    {
        return;
    }
    assert(a->data && a->data->data);
    assert(a->grad && a->grad->data && a->grad->data->data);

    ReductionExtras* reduction_extras = (ReductionExtras*) extras;
    int axis = reduction_extras->axis;

    int N = numel(a->shape, a->ndim);
    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    int *d_in_shape, *d_in_strides, *d_out_strides;
    copy_shape_and_strides_to_device(a->shape, a->strides, a->ndim, &d_in_shape, &d_in_strides);
    cudaMalloc(&d_out_strides, out->ndim * sizeof(int));
    cudaMemcpy(d_out_strides, out->strides, out->ndim * sizeof(int), cudaMemcpyHostToDevice);

    max_grad_kernel<<<num_blocks, num_threads_per_block>>>(
        out->grad->data->data, a->grad->data->data, a->data->data, out->data->data, N, d_in_shape,
        d_in_strides, a->ndim, d_out_strides, out->ndim, axis);

    cudaFree(d_in_shape);
    cudaFree(d_in_strides);
    cudaFree(d_out_strides);

    CHECK_CUDA();
}

