#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    void plast_cuda_abs_kernel_float(float* out, const float* in, size_t num_elements);
    void plast_cuda_abs_kernel_int32(int32_t* out, const int32_t* in, size_t num_elements);

    void plast_cuda_exp_kernel_float(float* out, const float* in, size_t num_elements);

    void plast_cuda_log_kernel_float(float* out, const float* in, size_t num_elements);

    void plast_cuda_relu_kernel_float(float* out, const float* in, size_t num_elements);

    void plast_cuda_leaky_relu_kernel_float(float* out, const float* in, size_t num_elements,
                                            float alpha);

#ifdef __cplusplus
}
#endif
