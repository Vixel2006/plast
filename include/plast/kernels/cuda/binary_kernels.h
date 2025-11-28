#pragma once

#include <stddef.h> // For size_t
#include <stdint.h> // For int32_t

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

#ifdef __cplusplus
}
#endif
