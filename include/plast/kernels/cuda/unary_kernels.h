#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    void plast_cuda_abs_kernel_float(float* out, const float* in, size_t num_elements);
    void plast_cuda_abs_kernel_int32(int32_t* out, const int32_t* in, size_t num_elements);
    void plast_cuda_abs_kernel_strided_float(float* out, const float* in, const size_t* out_shape,
                                             size_t out_ndim, const size_t* in_strides);
    void plast_cuda_abs_kernel_strided_int32(int32_t* out, const int32_t* in,
                                            const size_t* out_shape, size_t out_ndim,
                                            const size_t* in_strides);

    void plast_cuda_exp_kernel_float(float* out, const float* in, size_t num_elements);
    void plast_cuda_exp_kernel_strided_float(float* out, const float* in, const size_t* out_shape,
                                             size_t out_ndim, const size_t* in_strides);

    void plast_cuda_log_kernel_float(float* out, const float* in, size_t num_elements);
    void plast_cuda_log_kernel_strided_float(float* out, const float* in, const size_t* out_shape,
                                             size_t out_ndim, const size_t* in_strides);

    void plast_cuda_relu_kernel_float(float* out, const float* in, size_t num_elements);
    void plast_cuda_relu_kernel_strided_float(float* out, const float* in, const size_t* out_shape,
                                              size_t out_ndim, const size_t* in_strides);

    void plast_cuda_leaky_relu_kernel_float(float* out, const float* in, size_t num_elements,
                                            float alpha);
    void plast_cuda_leaky_relu_kernel_strided_float(float* out, const float* in,
                                                    const size_t* out_shape, size_t out_ndim,
                                                    const size_t* in_strides, float alpha);

#ifdef __cplusplus
}
#endif
