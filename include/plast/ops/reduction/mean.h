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

class MeanOperation : public BaseOperation
{
  public:
    MeanOperation(int dim, bool keepdim) : dim_(dim), keepdim_(keepdim), full_reduction_(false) {}
    MeanOperation(bool full_reduction) : full_reduction_(full_reduction) {}

    const std::string& name() const override
    {
        static std::string op_name = "mean";
        return op_name;
    }

    std::vector<size_t>
    infer_output_shape(const std::vector<std::vector<size_t>>& input_shapes) const override
    {
        if (full_reduction_)
        {
            return std::vector<size_t>(1, 1);
        }

        std::vector<size_t> output_shape = input_shapes[0];

        if (dim_ >= output_shape.size())
        {
            throw std::runtime_error("Can't reduce a tensor around a non-existing dimension.");
        }

        if (!keepdim_)
        {
            output_shape.erase(output_shape.begin() + dim_);
        }
        else
        {
            output_shape[dim_] = 1;
        }

        return output_shape;
    }

    tensor::Tensor execute_cpu(const std::vector<const tensor::Tensor*>& inputs) const override;
    tensor::Tensor execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const override;

  private:
    int dim_;
    bool keepdim_;
    bool full_reduction_;
};

} // namespace ops
} // namespace plast
