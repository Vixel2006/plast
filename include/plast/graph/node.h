#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "plast/core/types.h"
#include "plast/ops/base_op.h"
#include "plast/tensor/tensor.h"

namespace plast
{
namespace graph
{

class Node
{
  public:
    // Constructor for operation nodes
    Node(std::shared_ptr<ops::BaseOperation> op, const std::vector<std::shared_ptr<Node>>& inputs);
    // Constructor for leaf nodes (input tensors with actual values)
    Node(std::shared_ptr<tensor::Tensor> value);

    bool is_leaf() const { return op_ == nullptr; }
    const std::shared_ptr<ops::BaseOperation> operation() const { return op_; }
    const std::vector<std::shared_ptr<Node>>& inputs() const { return inputs_; }
    const std::vector<size_t>& shape() const;

    // For caching results during execution
    void set_output_tensor(std::shared_ptr<tensor::Tensor> value);
    std::shared_ptr<tensor::Tensor> get_output_tensor() const;
    bool has_output_tensor() const;
    void clear_output_tensor();

  private:
    std::shared_ptr<ops::BaseOperation> op_;
    std::vector<std::shared_ptr<Node>> inputs_;
    std::shared_ptr<tensor::Tensor>
        output_tensor_; // Stores the actual tensor value (for leaf nodes) or computed result
};

} // namespace graph
} // namespace plast
