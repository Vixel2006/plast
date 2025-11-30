#pragma once

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace plast
{
namespace core
{

// Function to broadcast two shapes and return the resulting broadcasted shape.
// Throws std::runtime_error if shapes are not broadcastable.
std::vector<size_t> broadcast_shapes(const std::vector<size_t>& shape1,
                                     const std::vector<size_t>& shape2);

// Function to calculate strides for a given shape.
// Strides indicate how many bytes to skip in memory to get to the next element along each
// dimension. For now, we'll calculate element strides (number of elements to skip).
std::vector<size_t> calculate_strides(const std::vector<size_t>& shape);

// Function to compute strides for a tensor given its original shape and a target (broadcasted)
// shape.
std::vector<size_t> compute_strides(const std::vector<size_t>& original_shape,
                                    const std::vector<size_t>& target_shape);

} // namespace core
} // namespace plast