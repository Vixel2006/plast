#pragma once

#include <stddef.h>
#include <stdint.h>
// #include "plast/core/types.h" // DType is no longer needed as types are explicit

#ifdef __cplusplus
extern "C"
{
#endif

// Full reduction kernels for float
void plast_cpu_min_full_reduction_float(const float* input_data, float* output_data,
                                  const size_t* input_shape, size_t input_ndim);

void plast_cpu_mean_full_reduction_float(const float* input_data, float* output_data,
                                   const size_t* input_shape, size_t input_ndim);

void plast_cpu_sum_full_reduction_float(const float* input_data, float* output_data,
                                  const size_t* input_shape, size_t input_ndim);

void plast_cpu_max_full_reduction_float(const float* input_data, float* output_data,
                                  const size_t* input_shape, size_t input_ndim);

// Full reduction kernels for int32
void plast_cpu_min_full_reduction_int32(const int32_t* input_data, int32_t* output_data,
                                  const size_t* input_shape, size_t input_ndim);

void plast_cpu_mean_full_reduction_int32(const int32_t* input_data, int32_t* output_data,
                                   const size_t* input_shape, size_t input_ndim);

void plast_cpu_sum_full_reduction_int32(const int32_t* input_data, int32_t* output_data,
                                  const size_t* input_shape, size_t input_ndim);

void plast_cpu_max_full_reduction_int32(const int32_t* input_data, int32_t* output_data,
                                  const size_t* input_shape, size_t input_ndim);

// Reduction along a dimension kernels for float
void plast_cpu_min_reduction_dim_float(const float* input_data, float* output_data,
                                 const size_t* input_shape, size_t input_ndim,
                                 const size_t* output_shape, size_t output_ndim,
                                 int dim);

void plast_cpu_mean_reduction_dim_float(const float* input_data, float* output_data,
                                  const size_t* input_shape, size_t input_ndim,
                                  const size_t* output_shape, size_t output_ndim,
                                  int dim);

void plast_cpu_sum_reduction_dim_float(const float* input_data, float* output_data,
                                 const size_t* input_shape, size_t input_ndim,
                                 const size_t* output_shape, size_t output_ndim,
                                 int dim);

void plast_cpu_max_reduction_dim_float(const float* input_data, float* output_data,
                                 const size_t* input_shape, size_t input_ndim,
                                 const size_t* output_shape, size_t output_ndim,
                                 int dim);

// Reduction along a dimension kernels for int32
void plast_cpu_min_reduction_dim_int32(const int32_t* input_data, int32_t* output_data,
                                 const size_t* input_shape, size_t input_ndim,
                                 const size_t* output_shape, size_t output_ndim,
                                 int dim);

void plast_cpu_mean_reduction_dim_int32(const int32_t* input_data, int32_t* output_data,
                                  const size_t* input_shape, size_t input_ndim,
                                  const size_t* output_shape, size_t output_ndim,
                                  int dim);

void plast_cpu_sum_reduction_dim_int32(const int32_t* input_data, int32_t* output_data,
                                 const size_t* input_shape, size_t input_ndim,
                                 const size_t* output_shape, size_t output_ndim,
                                 int dim);

void plast_cpu_max_reduction_dim_int32(const int32_t* input_data, int32_t* output_data,
                                 const size_t* input_shape, size_t input_ndim,
                                 const size_t* output_shape, size_t output_ndim,
                                 int dim);

#ifdef __cplusplus
}
#endif
