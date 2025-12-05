#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // CUDA kernel for element-wise addition of float tensors
    void plast_cuda_add_kernel_float(float* out, const float* in1, const float* in2,
                                     size_t num_elements);

    // CUDA kernel for element-wise addition of int32_t tensors
    void plast_cuda_add_kernel_int32(int32_t* out, const int32_t* in1, const int32_t* in2,
                                     size_t num_elements);

    // CUDA kernel for strided element-wise addition of float tensors
    void plast_cuda_add_kernel_strided_float(float* out, const float* in1, const float* in2,
                                             const size_t* out_shape, size_t out_ndim,
                                             const size_t* in1_strides, const size_t* in2_strides);

    // CUDA kernel for strided element-wise addition of int32_t tensors
    void plast_cuda_add_kernel_strided_int32(int32_t* out, const int32_t* in1, const int32_t* in2,
                                             const size_t* out_shape, size_t out_ndim,
                                             const size_t* in1_strides, const size_t* in2_strides);

    // CUDA kernel for element-wise subtraction of float tensors
    void plast_cuda_sub_kernel_float(float* out, const float* in1, const float* in2,
                                     size_t num_elements);

    // CUDA kernel for element-wise subtraction of int32_t tensors
    void plast_cuda_sub_kernel_int32(int32_t* out, const int32_t* in1, const int32_t* in2,
                                     size_t num_elements);

    // CUDA kernel for element-wise multiplication of float tensors
    void plast_cuda_mul_kernel_float(float* out, const float* in1, const float* in2,
                                     size_t num_elements);

    // CUDA kernel for element-wise multiplication of int32_t tensors
    void plast_cuda_mul_kernel_int32(int32_t* out, const int32_t* in1, const int32_t* in2,
                                     size_t num_elements);

    // Add declarations for other data types as needed

    // CUDA kernel for matrix multiplication of float tensors
    void plast_cuda_matmul_kernel_float(float* out, const float* in1, const float* in2, int B,
                                        int N, int M, int K);

    // CUDA kernel for matrix multiplication of int32_t tensors
    void plast_cuda_matmul_kernel_int32(int32_t* out, const int32_t* in1, const int32_t* in2, int B,
                                        int N, int M, int K);

    // CUDA kernel for matrix multiplication of float tensors with strided access
    void plast_cuda_matmul_kernel_strided_float(float* out, const float* in1, const float* in2,
                                                const size_t* out_shape, size_t out_ndim,
                                                const size_t* in1_strides, const size_t* in2_strides,
                                                size_t K_dim);

    // CUDA kernel for matrix multiplication of int32_t tensors with strided access
    void plast_cuda_matmul_kernel_strided_int32(int32_t* out, const int32_t* in1, const int32_t* in2,
                                                const size_t* out_shape, size_t out_ndim,
                                                const size_t* in1_strides, const size_t* in2_strides,
                                                size_t K_dim);

    // CUDA kernel for matrix multiplication of float tensors
    void plast_cuda_matmul_kernel_float(float* out, const float* in1, const float* in2, int B,
                                        int N, int M, int K);

#ifdef __cplusplus
}
#endif
