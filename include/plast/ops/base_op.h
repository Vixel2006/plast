#pragma once

#include <vector>
#include <string>
#include <memory> // For std::shared_ptr
#include <stdexcept> // For std::runtime_error

#include "plast/core/types.h"
#include "plast/tensor/tensor.h" // Include Tensor header

namespace plast {
namespace ops {

class BaseOperation {
public:
    virtual ~BaseOperation() = default;

    // Unique identifier for the operation type
    virtual const std::string& name() const = 0;

    // Infer output shape given input shapes
    virtual std::vector<size_t> infer_output_shape(const std::vector<std::vector<size_t>>& input_shapes) const = 0;

    // Execute the operation on CPU
    virtual tensor::Tensor execute_cpu(const std::vector<const tensor::Tensor*>& inputs) const = 0;

    // Execute the operation on CUDA
    virtual tensor::Tensor execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const = 0;

    // For autograd: compute gradients (placeholder for now)
    //virtual std::vector<tensor::Tensor> compute_gradients(...) = 0;
};

} // namespace ops
} // namespace plast
