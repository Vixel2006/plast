#include "plast/kernels/cuda/binary_kernels.h"
#include <stdexcept>

// Placeholder CUDA matmul kernel implementations
// TODO: Implement actual efficient CUDA matmul kernels

void plast_cuda_matmul_kernel_strided_float(float* out, const float* in1, const float* in2,
                                            const size_t* out_shape, size_t out_ndim,
                                            const size_t* in1_strides, const size_t* in2_strides,
                                            const size_t* in1_shape, const size_t* in2_shape,
                                            size_t K_dim)
{
    throw std::runtime_error("CUDA strided matmul kernel for float not yet implemented.");
}

void plast_cuda_matmul_kernel_strided_int32(int32_t* out, const int32_t* in1, const int32_t* in2,
                                            const size_t* out_shape, size_t out_ndim,
                                            const size_t* in1_strides, const size_t* in2_strides,
                                            const size_t* in1_shape, const size_t* in2_shape,
                                            size_t K_dim)
{
    throw std::runtime_error("CUDA strided matmul kernel for int32 not yet implemented.");
}
