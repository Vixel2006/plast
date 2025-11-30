#include "plast/graph/node.h"
#include <stdexcept>

namespace plast
{
namespace graph
{

// Constructor for operation nodes
Node::Node(std::shared_ptr<ops::BaseOperation> op, const std::vector<std::shared_ptr<Node>>& inputs)
    : op_(std::move(op)), inputs_(inputs), output_tensor_(nullptr) // Explicitly null for operation nodes initially
{
    if (!op_)
    {
        throw std::runtime_error("Operation cannot be null for an operation node.");
    }
    // Shape will be determined upon execution and stored in output_tensor_
}

// Constructor for leaf nodes (input tensors with actual values)
Node::Node(std::shared_ptr<tensor::Tensor> value)
    : op_(nullptr), inputs_({}), output_tensor_(std::move(value)) // Store the shared_ptr directly
{
    if (!output_tensor_) {
        throw std::runtime_error("Leaf node cannot be initialized with a null tensor.");
    }
}

const std::vector<size_t>& Node::shape() const
{
    if (!output_tensor_) {
        throw std::runtime_error("Attempted to access shape from a node with no output tensor. Node has not been executed or is not a leaf node.");
    }
    return output_tensor_->shape();
}

void Node::set_output_tensor(std::shared_ptr<tensor::Tensor> value)
{
    output_tensor_ = std::move(value);
}

std::shared_ptr<tensor::Tensor> Node::get_output_tensor() const
{
    if (!output_tensor_)
    {
        throw std::runtime_error(
            "Attempted to get output tensor from a node that has not been computed.");
    }
    return output_tensor_;
}

bool Node::has_output_tensor() const { return output_tensor_ != nullptr; }

void Node::clear_output_tensor() { output_tensor_.reset(); }

} // namespace graph
} // namespace plast