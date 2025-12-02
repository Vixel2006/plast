#pragma once

#include "plast/core/types.h"
#include "plast/ops/base_op.h"
#include "plast/tensor/tensor.h"

#include <numeric>
#include <string>
#include <vector>

namespace plast
{
namespace ops
{

class ExpandOperation : public BaseOperation
{
  public:
    ExpandOperation(const std::vector<size_t>& new_shape) : new_shape_(new_shape) {}

    const std::string& name() const override
    {
        static std::string op_name = "expand";
        return op_name;
    }

    std::vector<size_t>
    infer_output_shape(const std::vector<std::vector<size_t>>& input_shapes) const override
    {
        if (input_shapes.empty() || input_shapes[0].empty())
        {
            throw std::runtime_error("ExpandOperation expects at least one input shape.");
        }
        const std::vector<size_t>& input_shape = input_shapes[0];

        if (input_shape.size() != new_shape_.size())
        {
            throw std::runtime_error("ExpandOperation: New shape must have the same number of "
                                     "dimensions as input shape.");
        }

        for (size_t i = 0; i < input_shape.size(); ++i)
        {
            if (input_shape[i] != 1 && input_shape[i] != new_shape_[i])
            {
                throw std::runtime_error("ExpandOperation: Can only expand dimensions of size 1 or "
                                         "matching dimensions.");
            }
        }
        return new_shape_;
    }

    tensor::Tensor execute_cpu(const std::vector<const tensor::Tensor*>& inputs) const override;
    tensor::Tensor execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const override;

  private:
    std::vector<size_t> new_shape_;
};

} // namespace ops
} // namespace plast
