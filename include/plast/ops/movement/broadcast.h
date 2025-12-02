#pragma once

#include "plast/core/types.h"
#include "plast/ops/base_op.h"
#include "plast/tensor/tensor.h"

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

namespace plast
{
namespace ops
{

class BroadcastOperation : public BaseOperation
{
  public:
    BroadcastOperation(const std::vector<size_t>& target_shape) : target_shape_(target_shape) {}

    const std::string& name() const override
    {
        static std::string op_name = "broadcast";
        return op_name;
    }

    std::vector<size_t>
    infer_output_shape(const std::vector<std::vector<size_t>>& input_shapes) const override
    {
        if (input_shapes.empty() || input_shapes[0].empty())
        {
            throw std::runtime_error("BroadcastOperation expects at least one input shape.");
        }
        const std::vector<size_t>& input_shape = input_shapes[0];

        // Broadcasting rules:
        // 1. If the number of dimensions are different, prepend 1s to the smaller shape.
        // 2. Dimensions must either be equal, or one of them must be 1.
        // 3. The resulting dimension is the larger of the two.

        size_t max_dims = std::max(input_shape.size(), target_shape_.size());
        std::vector<size_t> padded_input_shape(max_dims);
        std::vector<size_t> padded_target_shape(max_dims);

        // Pad with 1s to the left
        std::fill(padded_input_shape.begin(), padded_input_shape.end() - input_shape.size(), 1);
        std::copy(input_shape.begin(), input_shape.end(),
                  padded_input_shape.begin() + (max_dims - input_shape.size()));

        std::fill(padded_target_shape.begin(), padded_target_shape.end() - target_shape_.size(), 1);
        std::copy(target_shape_.begin(), target_shape_.end(),
                  padded_target_shape.begin() + (max_dims - target_shape_.size()));

        std::vector<size_t> output_shape(max_dims);
        for (size_t i = 0; i < max_dims; ++i)
        {
            if (padded_input_shape[i] == padded_target_shape[i])
            {
                output_shape[i] = padded_input_shape[i];
            }
            else if (padded_input_shape[i] == 1)
            {
                output_shape[i] = padded_target_shape[i];
            }
            else if (padded_target_shape[i] == 1)
            {
                output_shape[i] = padded_input_shape[i];
            }
            else
            {
                throw std::runtime_error("BroadcastOperation: Shapes are not broadcastable.");
            }
        }
        return output_shape;
    }

    tensor::Tensor execute_cpu(const std::vector<const tensor::Tensor*>& inputs) const override;
    tensor::Tensor execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const override;

  private:
    std::vector<size_t> target_shape_;
};

} // namespace ops
} // namespace plast
