#pragma once

#include "plast/ops/base_op.h"
#include "plast/tensor/tensor.h"
#include "plast/core/types.h"
#include <string>
#include <vector>

namespace plast {
namespace ops {

class AddOperation : public BaseOperation {
public:
    const std::string& name() const override {
        static const std::string op_name = "Add";
        return op_name;
    }

    std::vector<size_t> infer_output_shape(const std::vector<std::vector<size_t>>& input_shapes) const override {
        if (input_shapes.size() != 2) {
            throw std::runtime_error("Add operation requires exactly two input tensors.");
        }
        // For simplicity, assume same shape for now.
        // Broadcasting logic would be more complex.
        if (input_shapes[0] != input_shapes[1]) {
            throw std::runtime_error("Add operation requires input tensors of the same shape (broadcasting not yet implemented).");
        }
        return input_shapes[0];
    }

    tensor::Tensor execute_cpu(const std::vector<const tensor::Tensor*>& inputs) const override;
    tensor::Tensor execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const override;
};

} // namespace ops
} // namespace plast
