#pragma once

#include "plast/core/types.h"
#include "plast/ops/base_op.h"
#include "plast/tensor/tensor.h"

#include <string>
#include <vector>

namespace plast
{
namespace ops
{

class ViewOperation : public BaseOperation
{
  public:
    ViewOperation(const std::vector<size_t>& new_shape) : new_shape_(new_shape) {}

    const std::string& name() const override
    {
        static std::string op_name = "view";
        return op_name;
    }

    std::vector<size_t>
    infer_output_shape(const std::vector<std::vector<size_t>>& input_shapes) const override
    {
        // For view, the output shape is the new_shape provided at construction
        return new_shape_;
    }

    tensor::Tensor execute_cpu(const std::vector<const tensor::Tensor*>& inputs) const override;
    tensor::Tensor execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const override;

  private:
    std::vector<size_t> new_shape_;
};

} // namespace ops
} // namespace plast
