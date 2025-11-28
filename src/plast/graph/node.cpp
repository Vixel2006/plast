#include "plast/graph/node.h"
#include <stdexcept>

namespace plast {
namespace graph {

// Constructor for operation nodes
Node::Node(std::shared_ptr<ops::BaseOperation> op, const std::vector<std::shared_ptr<Node>>& inputs)
    : op_(std::move(op)), inputs_(inputs) {
    if (!op_) {
        throw std::runtime_error("Operation cannot be null for an operation node.");
    }

    // Infer output shape from inputs
    std::vector<std::vector<size_t>> input_shapes;
    input_shapes.reserve(inputs_.size());
    for (const auto& input_node : inputs_) {
        input_shapes.push_back(input_node->shape());
    }
    shape_ = op_->infer_output_shape(input_shapes);
}

// Constructor for leaf nodes (input tensors with actual values)
Node::Node(const tensor::Tensor& value)
    : op_(nullptr), inputs_({}), cached_value_(value.clone()), shape_(value.shape()) {}

void Node::set_cached_value(tensor::Tensor&& value) {
    cached_value_ = std::move(value); // Store by moving
}

const tensor::Tensor& Node::get_cached_value() const {
    if (!cached_value_.has_value()) {
        throw std::runtime_error("Attempted to get cached value from a node that has not been computed.");
    }
    return *cached_value_; // Return a const reference
}

bool Node::has_cached_value() const {
    return cached_value_.has_value();
}

void Node::clear_cached_value() {
    cached_value_.reset();
}

} // namespace graph
} // namespace plast
