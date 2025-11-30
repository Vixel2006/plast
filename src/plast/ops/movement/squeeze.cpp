#include "plast/ops/movement/squeeze.h"

#include <numeric>

namespace plast
{
namespace ops
{

tensor::Tensor SqueezeOperation::execute_cpu(const std::vector<const tensor::Tensor*>& inputs) const
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("SqueezeOperation expects exactly one input tensor.");
    }

    const tensor::Tensor* input_tensor = inputs[0];
    std::vector<size_t> output_shape = input_tensor->shape();
    std::vector<size_t> output_strides = input_tensor->strides();

    if (N >= output_shape.size())
    {
        throw std::runtime_error("Squeeze dimension out of bounds.");
    }

    if (output_shape[N] == 1)
    {
        output_shape.erase(output_shape.begin() + N);
        output_strides.erase(output_strides.begin() + N);
    }
    else
    {
        return input_tensor->view(input_tensor->shape(), input_tensor->strides());
    }

        return input_tensor->reshape(output_shape, output_strides);}

tensor::Tensor
SqueezeOperation::execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("SqueezeOperation expects exactly one input tensor.");
    }

    const tensor::Tensor* input_tensor = inputs[0];
    std::vector<size_t> output_shape = input_tensor->shape();
    std::vector<size_t> output_strides = input_tensor->strides();

    if (N >= output_shape.size())
    {
        throw std::runtime_error("Squeeze dimension out of bounds.");
    }

    if (output_shape[N] == 1)
    {
        output_shape.erase(output_shape.begin() + N);
        output_strides.erase(output_strides.begin() + N);
    }
    else
    {
        return input_tensor->view(input_tensor->shape(), input_tensor->strides());
    }

        return input_tensor->reshape(output_shape, output_strides);}

} // namespace ops
} // namespace plast
