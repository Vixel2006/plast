#include "plast/ops/movement/unsqueeze.h"

#include <numeric>

namespace plast
{
namespace ops
{

tensor::Tensor
UnsqueezeOperation::execute_cpu(const std::vector<const tensor::Tensor*>& inputs) const
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("UnsqueezeOperation expects exactly one input tensor.");
    }

    const tensor::Tensor* input_tensor = inputs[0];
    std::vector<size_t> output_shape = input_tensor->shape();
    std::vector<size_t> output_strides = input_tensor->strides();

    if (dim_ > output_shape.size())
    {
        throw std::runtime_error("Unsqueeze dimension out of bounds.");
    }

    output_shape.insert(output_shape.begin() + dim_, 1);

    // Calculate new stride for the inserted dimension
    size_t new_stride;
    if (dim_ < output_strides.size())
    {
        new_stride = output_strides[dim_];
    }
    else
    {
        // If inserting at the end, the new stride is 1 (for a dimension of size 1)
        new_stride = 1;
    }
    output_strides.insert(output_strides.begin() + dim_, new_stride);

    // Create a new Tensor that views the same data but with new shape and strides
    return input_tensor->reshape(output_shape, output_strides);
}

tensor::Tensor
UnsqueezeOperation::execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("UnsqueezeOperation expects exactly one input tensor.");
    }

    const tensor::Tensor* input_tensor = inputs[0];
    std::vector<size_t> output_shape = input_tensor->shape();
    std::vector<size_t> output_strides = input_tensor->strides();

    if (dim_ > output_shape.size())
    {
        throw std::runtime_error("Unsqueeze dimension out of bounds.");
    }

    output_shape.insert(output_shape.begin() + dim_, 1);

    // Calculate new stride for the inserted dimension
    size_t new_stride;
    if (dim_ < output_strides.size())
    {
        new_stride = output_strides[dim_];
    }
    else
    {
        // If inserting at the end, the new stride is 1 (for a dimension of size 1)
        new_stride = 1;
    }
    output_strides.insert(output_strides.begin() + dim_, new_stride);

    // Create a new Tensor that views the same data but with new shape and strides
    return input_tensor->reshape(output_shape, output_strides);
}

} // namespace ops
} // namespace plast
