#include "plast/ops/movement/broadcast.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace plast
{
namespace ops
{

tensor::Tensor
BroadcastOperation::execute_cpu(const std::vector<const tensor::Tensor*>& inputs) const
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("BroadcastOperation expects exactly one input tensor.");
    }

    const tensor::Tensor* input_tensor = inputs[0];
    const std::vector<size_t>& input_shape = input_tensor->shape();
    const std::vector<size_t>& input_strides = input_tensor->strides();

    // Use the infer_output_shape logic to get the actual output shape and validate broadcastability
    std::vector<size_t> output_shape = infer_output_shape({input_shape});

    std::vector<size_t> output_strides(output_shape.size());

    size_t input_dims = input_shape.size();
    size_t output_dims = output_shape.size();

    // Calculate strides for broadcasting
    for (size_t i = 0; i < output_dims; ++i)
    {
        size_t input_padding_offset = output_dims - input_dims;

        if (i < input_padding_offset)
        {
            // This dimension was conceptually prepended to the input shape (broadcasted from 1)
            output_strides[i] = 0;
        }
        else
        {
            // This dimension corresponds to an actual dimension in the original input tensor
            size_t original_input_idx = i - input_padding_offset;

            if (input_shape[original_input_idx] == output_shape[i])
            {
                output_strides[i] = input_strides[original_input_idx];
            }
            else if (input_shape[original_input_idx] == 1)
            {
                output_strides[i] = 0; // Broadcast this dimension
            }
            else
            {
                // This case should ideally be caught by infer_output_shape, but as a safeguard
                throw std::runtime_error("BroadcastOperation: Internal error during stride "
                                         "calculation. Shapes not broadcastable.");
            }
        }
    }

    return input_tensor->view(output_shape, output_strides);
}

tensor::Tensor
BroadcastOperation::execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("BroadcastOperation expects exactly one input tensor.");
    }

    const tensor::Tensor* input_tensor = inputs[0];
    const std::vector<size_t>& input_shape = input_tensor->shape();
    const std::vector<size_t>& input_strides = input_tensor->strides();

    // Use the infer_output_shape logic to get the actual output shape and validate broadcastability
    std::vector<size_t> output_shape = infer_output_shape({input_shape});

    std::vector<size_t> output_strides(output_shape.size());

    size_t input_dims = input_shape.size();
    size_t output_dims = output_shape.size();

    // Calculate strides for broadcasting
    for (size_t i = 0; i < output_dims; ++i)
    {
        size_t input_padding_offset = output_dims - input_dims;

        if (i < input_padding_offset)
        {
            // This dimension was conceptually prepended to the input shape (broadcasted from 1)
            output_strides[i] = 0;
        }
        else
        {
            // This dimension corresponds to an actual dimension in the original input tensor
            size_t original_input_idx = i - input_padding_offset;

            if (input_shape[original_input_idx] == output_shape[i])
            {
                output_strides[i] = input_strides[original_input_idx];
            }
            else if (input_shape[original_input_idx] == 1)
            {
                output_strides[i] = 0; // Broadcast this dimension
            }
            else
            {
                // This case should ideally be caught by infer_output_shape, but as a safeguard
                throw std::runtime_error("BroadcastOperation: Internal error during stride "
                                         "calculation. Shapes not broadcastable.");
            }
        }
    }

    return input_tensor->view(output_shape, output_strides);
}

} // namespace ops
} // namespace plast
