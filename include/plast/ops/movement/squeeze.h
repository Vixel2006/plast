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

class SqueezeOperation : public BaseOperation
{
  public:
    SqueezeOperation(size_t N, size_t M) : N(N), M(M) {}

    const std::string& name() const override
    {
        static std::string op_name = "squeeze";
        return op_name;
    }

    std::vector<size_t>
    infer_output_shape(const std::vector<std::vector<size_t>>& input_shapes) const override
    {
        std::vector<size_t> output_shape = input_shapes[0];
        if (N >= output_shape.size())
        {
            throw std::runtime_error("Squeeze dimension out of bounds.");
        }
        if (output_shape[N] != 1)
        {
            // If the dimension is not 1, we don't squeeze it.
            // This might be an error or a no-op depending on desired behavior.
            // For now, we'll just return the original shape.
            return output_shape;
        }
        output_shape.erase(output_shape.begin() + N);
        return output_shape;
    }

    tensor::Tensor execute_cpu(const std::vector<const tensor::Tensor*>& inputs) const override;
    tensor::Tensor execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const override;

  private:
    size_t N, M;
    std::vector<size_t> new_shape_;
};

} // namespace ops
} // namespace plast
