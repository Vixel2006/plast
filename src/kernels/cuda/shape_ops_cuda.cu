#include "core/definitions.h"
#include "core/tensor.h"
#include "kernels/cuda/cuda_utils.cuh"

__global__ void reshape_backward_cuda_kernel(float *da, const float *dout, u64 num_elements) {
  u64 idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    da[idx] += dout[idx];
  }
}

__global__ void transpose_backward_cuda_kernel(float *da, const float *dout,
                                               const u64 *a_shape, const u64 *da_strides,
                                               const u64 *dout_strides, u64 ndim,
                                               u64 axis1, u64 axis2, u64 num_elements) {
  u64 idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    u64 coords[MAX_NDIM];
    // Convert linear index to coords in the space of a
    cuda_linear_to_coords(idx, a_shape, ndim, coords);
    u64 a_offset = cuda_get_offset(coords, da_strides, ndim);

    // Swap coordinates to find corresponding coordinate in output
    u64 coords_transposed[MAX_NDIM];
    for (u64 i = 0; i < ndim; ++i) {
      coords_transposed[i] = coords[i];
    }
    u64 temp = coords_transposed[axis1];
    coords_transposed[axis1] = coords_transposed[axis2];
    coords_transposed[axis2] = temp;

    u64 out_offset = cuda_get_offset(coords_transposed, dout_strides, ndim);
    atomicAdd(&da[a_offset], dout[out_offset]);
  }
}

extern "C" {

void reshape_backward_cuda(float *da, const float *dout, u64 num_elements) {
  u64 threads = 256;
  u64 blocks = (num_elements + threads - 1) / threads;
  reshape_backward_cuda_kernel<<<blocks, threads>>>(da, dout, num_elements);
  cudaDeviceSynchronize();
}

void transpose_backward_cuda(float *da, const float *dout,
                             const u64 *a_shape, const u64 *da_strides,
                             const u64 *dout_strides, u64 ndim,
                             u64 axis1, u64 axis2, u64 num_elements) {
  // Allocate shape and strides on GPU
  u64 *d_a_shape = NULL;
  u64 *d_da_strides = NULL;
  u64 *d_dout_strides = NULL;

  cudaMalloc(&d_a_shape, ndim * sizeof(u64));
  cudaMalloc(&d_da_strides, ndim * sizeof(u64));
  cudaMalloc(&d_dout_strides, ndim * sizeof(u64));

  cudaMemcpy(d_a_shape, a_shape, ndim * sizeof(u64), cudaMemcpyHostToDevice);
  cudaMemcpy(d_da_strides, da_strides, ndim * sizeof(u64), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dout_strides, dout_strides, ndim * sizeof(u64), cudaMemcpyHostToDevice);

  u64 threads = 256;
  u64 blocks = (num_elements + threads - 1) / threads;
  transpose_backward_cuda_kernel<<<blocks, threads>>>(da, dout, d_a_shape, d_da_strides, d_dout_strides, ndim, axis1, axis2, num_elements);
  cudaDeviceSynchronize();

  cudaFree(d_dout_strides);
  cudaFree(d_da_strides);
  cudaFree(d_a_shape);
}

}
