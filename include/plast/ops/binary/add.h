#pragma once

#include "plast/core/types.h"
#include "plast/ops/base_op.h"
#include "plast/tensor/tensor.h"
#include "plast/core/shape_utils_cpp.h" // For core::broadcast_shapes
#include <string>
#include <vector>

namespace plast
{
namespace ops
{

class AddOperation : public BaseOperation
{
  public:
    const std::string& name() const override
    {
        static const std::string op_name = "Add";
        return op_name;
    }

    std::vector<size_t>
    infer_output_shape(const std::vector<std::vector<size_t>>& input_shapes) const override
    {
        if (input_shapes.size() != 2)
        {
            throw std::runtime_error("Add operation requires exactly two input tensors.");
        }
        return core::broadcast_shapes(input_shapes[0], input_shapes[1]);
    }

    tensor::Tensor execute_cpu(const std::vector<const tensor::Tensor*>& inputs) const override;
    tensor::Tensor execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const override;
};

} // namespace ops
} // namespace plast
