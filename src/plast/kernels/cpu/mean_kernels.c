#include "plast/kernels/cpu/reduction_kernels.h"
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

// Helper function to calculate total number of elements
static size_t calculate_total_elements(const size_t* shape, size_t ndim)
{
    size_t total = 1;
    for (size_t i = 0; i < ndim; ++i)
    {
        total *= shape[i];
    }
    return total;
}

// Helper function to calculate strides
// NOTE: This function assumes row-major order (C-style)
static void calculate_strides(const size_t* shape, size_t ndim, size_t* strides)
{
    if (ndim == 0) return;
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i)
    {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

// --- Full Reduction Kernels (float) ---

void plast_cpu_mean_full_reduction_float(const float* input_data, float* output_data,
                                         const size_t* input_shape, size_t input_ndim)
{
    size_t total_elements = calculate_total_elements(input_shape, input_ndim);
    if (total_elements == 0)
    {
        output_data[0] = NAN;
        return;
    }

    double sum = 0.0; // Use double for sum to maintain precision
    for (size_t i = 0; i < total_elements; ++i)
    {
        sum += input_data[i];
    }
    output_data[0] = (float) (sum / total_elements);
}

// --- Full Reduction Kernels (int32) ---

void plast_cpu_mean_full_reduction_int32(const int32_t* input_data, int32_t* output_data,
                                         const size_t* input_shape, size_t input_ndim)
{
    size_t total_elements = calculate_total_elements(input_shape, input_ndim);
    if (total_elements == 0)
    {
        output_data[0] = 0;
        return;
    }

    int64_t sum = 0; // Use int64_t for sum to prevent overflow
    for (size_t i = 0; i < total_elements; ++i)
    {
        sum += input_data[i];
    }
    // Integer division for mean of integers
    output_data[0] = (int32_t) (sum / total_elements);
}

// --- Reduction along a dimension kernels (float) ---

void plast_cpu_mean_reduction_dim_float(const float* input_data, float* output_data,
                                        const size_t* input_shape, size_t input_ndim,
                                        const size_t* output_shape, size_t output_ndim, int dim)
{
    size_t input_strides[input_ndim];
    calculate_strides(input_shape, input_ndim, input_strides);

    size_t output_strides[output_ndim];
    calculate_strides(output_shape, output_ndim, output_strides);

    size_t reduction_size = input_shape[dim];

    size_t output_total_elements = calculate_total_elements(output_shape, output_ndim);

    for (size_t out_flat_idx = 0; out_flat_idx < output_total_elements; ++out_flat_idx)
    {
        size_t current_input_base_idx = 0;
        size_t temp_out_flat_idx = out_flat_idx;
        size_t current_output_dim_idx = 0;

        for (size_t i = 0; i < input_ndim; ++i)
        {
            if (i == dim)
            {
                // Skip the reduced dimension for base index calculation
            }
            else
            {
                size_t coord_in_this_dim =
                    (temp_out_flat_idx / output_strides[current_output_dim_idx]) %
                    output_shape[current_output_dim_idx];
                current_input_base_idx += coord_in_this_dim * input_strides[i];
                current_output_dim_idx++;
            }
        }

        double current_sum = 0.0;

        for (size_t k = 0; k < reduction_size; ++k)
        {
            size_t input_idx = current_input_base_idx + k * input_strides[dim];
            current_sum += input_data[input_idx];
        }
        output_data[out_flat_idx] = (float) (current_sum / reduction_size);
    }
}

// --- Reduction along a dimension kernels (int32) ---

void plast_cpu_mean_reduction_dim_int32(const int32_t* input_data, int32_t* output_data,
                                        const size_t* input_shape, size_t input_ndim,
                                        const size_t* output_shape, size_t output_ndim, int dim)
{
    size_t input_strides[input_ndim];
    calculate_strides(input_shape, input_ndim, input_strides);

    size_t output_strides[output_ndim];
    calculate_strides(output_shape, output_ndim, output_strides);

    size_t reduction_size = input_shape[dim];

    size_t output_total_elements = calculate_total_elements(output_shape, output_ndim);

    for (size_t out_flat_idx = 0; out_flat_idx < output_total_elements; ++out_flat_idx)
    {
        size_t current_input_base_idx = 0;
        size_t temp_out_flat_idx = out_flat_idx;
        size_t current_output_dim_idx = 0;

        for (size_t i = 0; i < input_ndim; ++i)
        {
            if (i == dim)
            {
                // Skip the reduced dimension for base index calculation
            }
            else
            {
                size_t coord_in_this_dim =
                    (temp_out_flat_idx / output_strides[current_output_dim_idx]) %
                    output_shape[current_output_dim_idx];
                current_input_base_idx += coord_in_this_dim * input_strides[i];
                current_output_dim_idx++;
            }
        }

        int64_t current_sum = 0;

        for (size_t k = 0; k < reduction_size; ++k)
        {
            size_t input_idx = current_input_base_idx + k * input_strides[dim];
            current_sum += input_data[input_idx];
        }
        output_data[out_flat_idx] = (int32_t) (current_sum / reduction_size);
    }
}
