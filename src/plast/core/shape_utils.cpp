#include "plast/core/shape_utils_c.h"
#include "plast/core/shape_utils_cpp.h"
#include <numeric>
#include <stdexcept>

namespace plast
{
namespace core
{

std::vector<size_t> broadcast_shapes(const std::vector<size_t>& shape1,
                                     const std::vector<size_t>& shape2)
{
    size_t ndim1 = shape1.size();
    size_t ndim2 = shape2.size();
    size_t output_ndim = std::max(ndim1, ndim2);
    std::vector<size_t> output_shape(output_ndim);

    for (size_t i = 0; i < output_ndim; ++i)
    {
        size_t dim1 = (i < ndim1) ? shape1[ndim1 - 1 - i] : 1;
        size_t dim2 = (i < ndim2) ? shape2[ndim2 - 1 - i] : 1;

        if (dim1 != dim2 && dim1 != 1 && dim2 != 1)
        {
            throw std::runtime_error("Shapes are not broadcastable.");
        }
        output_shape[output_ndim - 1 - i] = std::max(dim1, dim2);
    }
    return output_shape;
}

std::vector<size_t> calculate_strides(const std::vector<size_t>& shape)
{
    size_t ndim = shape.size();
    std::vector<size_t> strides(ndim);
    if (ndim == 0)
    {
        return strides;
    }

    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i)
    {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

std::vector<size_t> compute_strides(const std::vector<size_t>& original_shape,
                                    const std::vector<size_t>& target_shape)
{
    size_t original_ndim = original_shape.size();
    size_t target_ndim = target_shape.size();
    std::vector<size_t> strides(target_ndim);

    // Calculate strides for the original shape
    std::vector<size_t> original_strides = calculate_strides(original_shape);

    // Iterate through the dimensions of the target shape (from right to left)
    for (int i = target_ndim - 1; i >= 0; --i)
    {
        // Corresponding dimension in the original shape
        int original_dim_idx = original_ndim - 1 - (target_ndim - 1 - i);

        if (original_dim_idx >= 0)
        {
            // If the dimension exists in the original shape
            if (original_shape[original_dim_idx] == target_shape[i])
            {
                // If dimensions match, use the original stride
                strides[i] = original_strides[original_dim_idx];
            }
            else if (original_shape[original_dim_idx] == 1)
            {
                // If original dimension is 1, it's broadcasted, so stride is 0
                strides[i] = 0;
            }
            else
            {
                // This case should ideally not happen if broadcast_shapes was called first
                throw std::runtime_error("Error in compute_strides: non-broadcastable dimension.");
            }
        }
        else
        {
            // If the dimension does not exist in the original shape (prepended 1s), stride is 0
            strides[i] = 0;
        }
    }
    return strides;
}

} // namespace core
} // namespace plast

// C-compatible implementations
size_t get_index(const size_t* current_indices, const size_t* strides, size_t ndim)
{
    size_t index = 0;
    for (size_t i = 0; i < ndim; ++i)
    {
        // If stride is 0, it means this dimension was broadcasted from size 1.
        // In this case, the index for this dimension should effectively be 0,
        // so we don't multiply current_indices[i] by strides[i].
        // The element at index 0 of the original dimension is always accessed.
        if (strides[i] != 0) {
            index += current_indices[i] * strides[i];
        }
    }
    return index;
}

void increment_indices(size_t* current_indices, const size_t* shape, size_t ndim)
{
    for (int i = ndim - 1; i >= 0; --i)
    {
        current_indices[i]++;
        if (current_indices[i] < shape[i])
        {
            break;
        }
        else
        {
            current_indices[i] = 0;
        }
    }
}
