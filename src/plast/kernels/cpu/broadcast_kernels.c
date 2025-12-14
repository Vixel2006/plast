#include "plast/kernels/cpu/broadcast_kernels.h"
#include <stddef.h> // For size_t
#include <string.h> // For memcpy

// Helper to calculate the linear index in a multi-dimensional array given coordinates and strides
static size_t calculate_linear_index(const size_t* coords, const size_t* strides, size_t ndim) {
    size_t index = 0;
    for (size_t i = 0; i < ndim; ++i) {
        index += coords[i] * strides[i];
    }
    return index;
}

// Helper to increment multi-dimensional coordinates
static void increment_coords(size_t* coords, const size_t* shape, size_t ndim) {
    for (int i = ndim - 1; i >= 0; --i) {
        coords[i]++;
        if (coords[i] < shape[i]) {
            return;
        }
        coords[i] = 0;
    }
}

void cpu_broadcast_kernel(const void* input_data,
                          const size_t* input_shape,
                          const size_t* input_strides,
                          size_t input_ndim,
                          void* output_data,
                          const size_t* output_shape,
                          size_t output_ndim,
                          size_t item_size) {
    char* src = (char*)input_data;
    char* dst = (char*)output_data;

    // Calculate total number of elements in the output tensor
    size_t total_output_elements = 1;
    for (size_t i = 0; i < output_ndim; ++i) {
        total_output_elements *= output_shape[i];
    }

    // Create padded input shape and strides for easier index calculation
    size_t padded_input_shape[output_ndim];
    size_t padded_input_strides[output_ndim];

    size_t input_padding_offset = output_ndim - input_ndim;

    for (size_t i = 0; i < output_ndim; ++i) {
        if (i < input_padding_offset) {
            // Dimensions prepended to input shape are treated as size 1 for indexing
            padded_input_shape[i] = 1;
            padded_input_strides[i] = 0; // Stride for broadcasted dimension
        } else {
            size_t original_input_idx = i - input_padding_offset;
            padded_input_shape[i] = input_shape[original_input_idx];
            if (input_shape[original_input_idx] == 1 && output_shape[i] > 1) {
                padded_input_strides[i] = 0; // Broadcast this dimension
            } else {
                padded_input_strides[i] = input_strides[original_input_idx];
            }
        }
    }

    // Iterate over all elements of the output tensor
    size_t output_coords[output_ndim];
    for (size_t i = 0; i < output_ndim; ++i) {
        output_coords[i] = 0;
    }
    for (size_t i = 0; i < total_output_elements; ++i) {
        // Calculate the source index based on the output coordinates and padded input strides
        size_t src_linear_index = calculate_linear_index(output_coords, padded_input_strides, output_ndim);
        
        // Copy the data
        memcpy(dst + i * item_size, src + src_linear_index * item_size, item_size);

        // Increment output coordinates
        increment_coords(output_coords, output_shape, output_ndim);
    }
}
