#include "plast/execution/engine.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace plast
{
namespace execution
{

ExecutionEngine::ExecutionEngine() {}

ExecutionEngine::~ExecutionEngine() {}

void ExecutionEngine::visit(std::shared_ptr<graph::Node> node,
                            std::unordered_map<std::shared_ptr<graph::Node>, bool>& visited,
                            std::unordered_map<std::shared_ptr<graph::Node>, bool>& in_stack,
                            std::vector<std::shared_ptr<graph::Node>>& sorted_nodes)
{
    visited[node] = true;
    in_stack[node] = true;

    for (const auto& input_node : node->inputs())
    {
        if (!visited[input_node])
        {
            visit(input_node, visited, in_stack, sorted_nodes);
        }
        else if (in_stack[input_node])
        {
            throw std::runtime_error("Cycle detected in computation graph!");
        }
    }

    in_stack[node] = false;
    sorted_nodes.push_back(node);
}

std::vector<std::shared_ptr<graph::Node>>
ExecutionEngine::topological_sort(std::shared_ptr<graph::Node> root_node)
{
    std::vector<std::shared_ptr<graph::Node>> sorted_nodes;
    std::unordered_map<std::shared_ptr<graph::Node>, bool> visited;
    std::unordered_map<std::shared_ptr<graph::Node>, bool> in_stack;

    visit(root_node, visited, in_stack, sorted_nodes);
    return sorted_nodes;
}

std::shared_ptr<plast::tensor::Tensor>
ExecutionEngine::execute(std::shared_ptr<graph::Node> root_node)
{
    if (!root_node)
    {
        throw std::runtime_error("Cannot execute a null graph node.");
    }

    std::vector<std::shared_ptr<graph::Node>> sorted_nodes = topological_sort(root_node);

    // Clear output tensors for all NON-LEAF nodes in the current graph before execution
    for (const auto& node : sorted_nodes)
    {
        if (!node->is_leaf())
        {
            node->clear_output_tensor();
        }
    }

    for (const auto& node : sorted_nodes)
    {
        if (node->is_leaf())
        {
            if (!node->has_output_tensor())
            {
                throw std::runtime_error(
                    "Leaf node without output tensor encountered during execution.");
            }
            continue;
        }

        // Collect inputs for the current operation
        std::vector<const tensor::Tensor*> inputs_for_op;
        for (const auto& input_node : node->inputs())
        {
            if (!input_node->has_output_tensor())
            {
                throw std::runtime_error(
                    "Input node value not computed before its dependent operation.");
            }
            inputs_for_op.push_back(input_node->get_output_tensor().get());
        }

        // Execute the operation
        if (inputs_for_op.empty())
        {
            throw std::runtime_error("Operation with no tensor inputs not yet supported.");
        }

        // Determine target device for execution
        core::DeviceType target_device =
            inputs_for_op[0]->device(); // Simple heuristic: use first input's device
        for (size_t i = 1; i < inputs_for_op.size(); ++i)
        {
            if (inputs_for_op[i]->device() != target_device)
            {
                throw std::runtime_error("Inputs to an operation are on different devices. "
                                         "Automatic transfer not yet implemented.");
            }
        }

        if (target_device == core::DeviceType::CPU)
        {
            tensor::Tensor output_tensor = node->operation()->execute_cpu(inputs_for_op);
            node->set_output_tensor(std::make_shared<plast::tensor::Tensor>(
                std::move(output_tensor))); // Cache the result
        }
        else if (target_device == core::DeviceType::CUDA)
        {
            tensor::Tensor output_tensor = node->operation()->execute_cuda(inputs_for_op);
            node->set_output_tensor(std::make_shared<plast::tensor::Tensor>(
                std::move(output_tensor))); // Cache the result
        }
        else
        {
            throw std::runtime_error("Unsupported device type for operation execution.");
        }
    }

    // The result of the root node is the final output
    if (!root_node->has_output_tensor())
    {
        throw std::runtime_error("Root node value not computed after graph execution.");
    }
    return root_node->get_output_tensor(); // Return the shared_ptr
}

} // namespace execution
} // namespace plast
