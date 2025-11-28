#pragma once

#include <memory>
#include <string>
#include <vector>
#include <optional>

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
    Node(const tensor::Tensor& value);

    bool is_leaf() const { return op_ == nullptr; }
    const std::shared_ptr<ops::BaseOperation> operation() const { return op_; }
    const std::vector<std::shared_ptr<Node>>& inputs() const { return inputs_; }
    const std::vector<size_t>& shape() const { return shape_; }

    // For caching results during execution
    void set_cached_value(tensor::Tensor&& value);
    const tensor::Tensor& get_cached_value() const;
    bool has_cached_value() const;
    void clear_cached_value();

  private:
    std::shared_ptr<ops::BaseOperation> op_;
    std::vector<std::shared_ptr<Node>> inputs_;
    std::optional<tensor::Tensor> cached_value_; // Store computed result
    std::vector<size_t> shape_;
};

} // namespace graph
} // namespace plast
