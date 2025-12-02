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

class UnsqueezeOperation : public BaseOperation
{
  public:
    UnsqueezeOperation(size_t dim) : dim_(dim) {}

    const std::string& name() const override
    {
        static std::string op_name = "unsqueeze";
        return op_name;
    }

    std::vector<size_t>
    infer_output_shape(const std::vector<std::vector<size_t>>& input_shapes) const override
    {
        std::vector<size_t> output_shape = input_shapes[0];
        if (dim_ > output_shape.size())
        {
            throw std::runtime_error("Unsqueeze dimension out of bounds.");
        }
        output_shape.insert(output_shape.begin() + dim_, 1);
        return output_shape;
    }

    tensor::Tensor execute_cpu(const std::vector<const tensor::Tensor*>& inputs) const override;
    tensor::Tensor execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const override;

  private:
    size_t dim_;
};

} // namespace ops
} // namespace plast
