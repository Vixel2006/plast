#include "autograd/cuda/reduction/common.cuh"
#include "autograd/cuda/reduction/reduction_ops_cuda.h"
#include "device_management.h"
#include "utils/indexing.cuh"

__global__ void sum_full_grad_kernel(float* in_grad_data, const float* output_grad, int in_size,
                                     const int* in_grad_shape, const int* in_grad_strides,
                                     int in_grad_ndim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < in_size; i += stride)
    {
        int in_grad_idx = get_idx(in_grad_shape, in_grad_strides, in_grad_ndim, i);
        in_grad_data[in_grad_idx] += output_grad[0];
    }
}

void sum_full_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("sum_full_grad_op_cuda: Entering function with n_prev=%d", n_prev);

    assert(out && out->grad && out->grad->data && out->grad->data->data);
    assert(prev && n_prev == 1 && prev[0]);

    Tensor* a = prev[0];
    if (!a->requires_grad)
    {
        return;
    }
    assert(a->grad && a->grad->data && a->grad->data->data);

    int N = numel(a->shape, a->ndim);
    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    if (is_contiguous(a->grad))
    {
        // A simplified kernel could be used here, but the generic one also works
        // for contiguous tensors if strides are correct.
        // For simplicity, we use the generic one.
    }

    int *d_in_grad_shape, *d_in_grad_strides;
    copy_shape_and_strides_to_device(a->grad->shape, a->grad->strides, a->grad->ndim,
                                     &d_in_grad_shape, &d_in_grad_strides);

    sum_full_grad_kernel<<<num_blocks, num_threads_per_block>>>(
        a->grad->data->data, out->grad->data->data, N, d_in_grad_shape, d_in_grad_strides,
        a->grad->ndim);

    cudaFree(d_in_grad_shape);
    cudaFree(d_in_grad_strides);

    CHECK_CUDA();

    LOG_INFO("sum_full_grad_op_cuda: Exiting function.");
}
