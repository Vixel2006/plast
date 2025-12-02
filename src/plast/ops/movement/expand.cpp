#include "plast/ops/movement/expand.h"

#include <numeric>
#include <stdexcept>

namespace plast
{
namespace ops
{

tensor::Tensor ExpandOperation::execute_cpu(const std::vector<const tensor::Tensor*>& inputs) const
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("ExpandOperation expects exactly one input tensor.");
    }

    const tensor::Tensor* input_tensor = inputs[0];
    const std::vector<size_t>& input_shape = input_tensor->shape();
    const std::vector<size_t>& input_strides = input_tensor->strides();

    if (input_shape.size() != new_shape_.size())
    {
        throw std::runtime_error(
            "ExpandOperation: New shape must have the same number of dimensions as input shape.");
    }

    std::vector<size_t> output_strides(new_shape_.size());
    for (size_t i = 0; i < new_shape_.size(); ++i)
    {
        if (input_shape[i] == 1 && new_shape_[i] > 1)
        {
            output_strides[i] = 0; // Broadcast this dimension
        }
        else if (input_shape[i] == new_shape_[i])
        {
            output_strides[i] = input_strides[i];
        }
        else
        {
            throw std::runtime_error(
                "ExpandOperation: Cannot expand non-singleton dimension to a different size.");
        }
    }

    return input_tensor->reshape(new_shape_, output_strides);
}

tensor::Tensor ExpandOperation::execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("ExpandOperation expects exactly one input tensor.");
    }

    const tensor::Tensor* input_tensor = inputs[0];
    const std::vector<size_t>& input_shape = input_tensor->shape();
    const std::vector<size_t>& input_strides = input_tensor->strides();

    if (input_shape.size() != new_shape_.size())
    {
        throw std::runtime_error(
            "ExpandOperation: New shape must have the same number of dimensions as input shape.");
    }

    std::vector<size_t> output_strides(new_shape_.size());
    for (size_t i = 0; i < new_shape_.size(); ++i)
    {
        if (input_shape[i] == 1 && new_shape_[i] > 1)
        {
            output_strides[i] = 0; // Broadcast this dimension
        }
        else if (input_shape[i] == new_shape_[i])
        {
            output_strides[i] = input_strides[i];
        }
        else
        {
            throw std::runtime_error(
                "ExpandOperation: Cannot expand non-singleton dimension to a different size.");
        }
    }

    return input_tensor->reshape(new_shape_, output_strides);
}

} // namespace ops
} // namespace plast
