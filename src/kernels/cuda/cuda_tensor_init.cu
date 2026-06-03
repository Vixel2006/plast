#include "cuda_tensor_init.h"
#include "definitions.h"
#include <cuda_runtime.h>

__global__ void zeros_kernel_int32(i32 *data, u64 num_elements) {
  u64 idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    data[idx] = 0;
  }
}

__global__ void zeros_kernel_float32(float *data, u64 num_elements) {
  u64 idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    data[idx] = 0.0f;
  }
}

void zeros_cuda(Tensor *t, u64 num_elements) {
  // Determine grid and block dimensions
  u64 threadsPerBlock = 256;
  u64 blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

  switch (t->dtype) {
  case INT32: {
    zeros_kernel_int32<<<blocksPerGrid, threadsPerBlock>>>((i32 *)t->data, num_elements);
    break;
  }
  case FLOAT32: {
    zeros_kernel_float32<<<blocksPerGrid, threadsPerBlock>>>((float *)t->data, num_elements);
    break;
  }
  default:
    // Handle unsupported dtype or error
    break;
  }
  cudaDeviceSynchronize(); // Ensure kernel completes
}

__global__ void ones_kernel_int32(i32 *data, u64 num_elements) {
  u64 idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    data[idx] = 1;
  }
}

__global__ void ones_kernel_float32(float *data, u64 num_elements) {
  u64 idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    data[idx] = 1.0f;
  }
}

void ones_cuda(Tensor *t, u64 num_elements) {
  // Determine grid and block dimensions
  u64 threadsPerBlock = 256;
  u64 blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

  switch (t->dtype) {
  case INT32: {
    ones_kernel_int32<<<blocksPerGrid, threadsPerBlock>>>((i32 *)t->data, num_elements);
    break;
  }
  case FLOAT32: {
    ones_kernel_float32<<<blocksPerGrid, threadsPerBlock>>>((float *)t->data, num_elements);
    break;
  }
  default:
    // Handle unsupported dtype or error
    break;
  }
  cudaDeviceSynchronize(); // Ensure kernel completes
}

void set_ones_grad_cuda(Tensor *t) {
  u64 num_elements = numel(t); // Assuming numel is accessible or passed
  // Determine grid and block dimensions
  u64 threadsPerBlock = 256;
  u64 blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

  switch (t->dtype) {
  case INT32: {
    ones_kernel_int32<<<blocksPerGrid, threadsPerBlock>>>((i32 *)t->grad->data, num_elements);
    break;
  }
  case FLOAT32: {
    ones_kernel_float32<<<blocksPerGrid, threadsPerBlock>>>((float *)t->grad->data, num_elements);
    break;
  }
  default:
    // Handle unsupported dtype or error
    break;
  }
  cudaDeviceSynchronize(); // Ensure kernel completes
}
