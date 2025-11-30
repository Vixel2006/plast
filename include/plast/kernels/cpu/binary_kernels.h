#pragma once

#include <stddef.h> // For size_t
#include <stdint.h> // For int32_t

#ifdef __cplusplus
extern "C"
{
#endif

    // CPU kernel for element-wise addition of float tensors
    void plast_cpu_add_kernel_float(float* out, const float* in1, const float* in2,
                                    size_t num_elements);

    // CPU kernel for element-wise addition of int32_t tensors
    void plast_cpu_add_kernel_int32(int32_t* out, const int32_t* in1, const int32_t* in2,
                                    size_t num_elements);

    // CPU kernel for element-wise subtraction of float tensors
    void plast_cpu_sub_kernel_float(float* out, const float* in1, const float* in2,
                                    size_t num_elements);

    // CPU kernel for element-wise subtraction of int32_t tensors
    void plast_cpu_sub_kernel_int32(int32_t* out, const int32_t* in1, const int32_t* in2,
                                    size_t num_elements);

    // CPU kernel for element-wise multiplication of float tensors
    void plast_cpu_mul_kernel_float(float* out, const float* in1, const float* in2,
                                    size_t num_elements);

    // CPU kernel for element-wise multipliation of int32_t tensors
    void plast_cpu_mul_kernel_int32(int32_t* out, const int32_t* in1, const int32_t* in2,
                                    size_t num_elements);

    void plast_cpu_matmul_kernel_float(float* out, const float* in1, const float* in2, const int B,
                                       const int N, const int M, const int K);

    void plast_cpu_matmul_kernel_int32(int32_t* out, const int32_t* in1, const int32_t* in2,
                                       const int B, const int N, const int M, const int K);

    // CPU kernel for matrix multiplication of float tensors with strided access
    void plast_cpu_matmul_kernel_strided_float(float* out, const float* in1, const float* in2,
                                               const size_t* out_shape, size_t out_ndim,
                                               const size_t* in1_strides,
                                               const size_t* in2_strides,
                                               const size_t* in1_shape, const size_t* in2_shape);

    // CPU kernel for matrix multiplication of int32_t tensors with strided access
    void plast_cpu_matmul_kernel_strided_int32(int32_t* out, const int32_t* in1,
                                               const int32_t* in2, const size_t* out_shape,
                                               size_t out_ndim, const size_t* in1_strides,
                                               const size_t* in2_strides,
                                               const size_t* in1_shape, const size_t* in2_shape);

    // CPU kernel for element-wise addition of float tensors with strided access
    void plast_cpu_add_kernel_strided_float(float* out, const float* in1, const float* in2,
                                            const size_t* out_shape, size_t out_ndim,
                                            const size_t* in1_strides, const size_t* in2_strides);

    // CPU kernel for element-wise addition of int32_t tensors with strided access
    void plast_cpu_add_kernel_strided_int32(int32_t* out, const int32_t* in1, const int32_t* in2,
                                            const size_t* out_shape, size_t out_ndim,
                                            const size_t* in1_strides, const size_t* in2_strides);

    // CPU kernel for element-wise subtraction of float tensors with strided access
    void plast_cpu_sub_kernel_strided_float(float* out, const float* in1, const float* in2,
                                            const size_t* out_shape, size_t out_ndim,
                                            const size_t* in1_strides, const size_t* in2_strides);

    // CPU kernel for element-wise subtraction of int32_t tensors with strided access
    void plast_cpu_sub_kernel_strided_int32(int32_t* out, const int32_t* in1, const int32_t* in2,
                                            const size_t* out_shape, size_t out_ndim,
                                            const size_t* in1_strides, const size_t* in2_strides);

    // CPU kernel for element-wise multiplication of float tensors with strided access
    void plast_cpu_mul_kernel_strided_float(float* out, const float* in1, const float* in2,
                                            const size_t* out_shape, size_t out_ndim,
                                            const size_t* in1_strides, const size_t* in2_strides);

    // CPU kernel for element-wise multiplication of int32_t tensors with strided access
    void plast_cpu_mul_kernel_strided_int32(int32_t* out, const int32_t* in1, const int32_t* in2,
                                            const size_t* out_shape, size_t out_ndim,
                                            const size_t* in1_strides, const size_t* in2_strides);

#ifdef __cplusplus
}
#endif
