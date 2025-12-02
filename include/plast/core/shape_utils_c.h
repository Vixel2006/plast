#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // Function to get the linear index from multi-dimensional indices and strides.
    size_t get_index(const size_t* current_indices, const size_t* strides, size_t ndim);

    // Function to increment multi-dimensional indices based on the shape.
    void increment_indices(size_t* current_indices, const size_t* shape, size_t ndim);

#ifdef __cplusplus
}
#endif
