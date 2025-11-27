#include "plast/execution/engine.h"
#include <algorithm> // For std::reverse
#include <iostream>  // For debugging
#include <stdexcept> // For std::runtime_error

namespace plast
{
namespace execution
{

ExecutionEngine::ExecutionEngine()
{
    // Constructor implementation if needed
}

ExecutionEngine::~ExecutionEngine()
{
    // Destructor implementation if needed
}

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
    std::reverse(sorted_nodes.begin(), sorted_nodes.end());
    return sorted_nodes;
}

tensor::Tensor ExecutionEngine::execute(std::shared_ptr<graph::Node> root_node)
{
    if (!root_node)
    {
        throw std::runtime_error("Cannot execute a null graph node.");
    }

    // Clear cache before new execution if not already cleared
    clear_cache();

    std::vector<std::shared_ptr<graph::Node>> sorted_nodes = topological_sort(root_node);

    for (const auto& node : sorted_nodes)
    {
        if (node->is_leaf())
        {
            // Leaf nodes already have their value set in the constructor
            // No computation needed, just ensure it's cached
            if (!node->has_cached_value())
            {
                throw std::runtime_error(
                    "Leaf node without cached value encountered during execution.");
            }
            continue;
        }

        // Collect inputs for the current operation
        std::vector<const tensor::Tensor*> inputs_for_op;
        for (const auto& input_node : node->inputs())
        {
            if (!input_node->has_cached_value())
            {
                throw std::runtime_error(
                    "Input node value not computed before its dependent operation.");
            }
            inputs_for_op.push_back(&input_node->get_cached_value());
        }

        // Execute the operation
        if (inputs_for_op.empty())
        {
            // Handle operations with no tensor inputs (e.g., constant creation, random init)
            // For now, assume all ops have tensor inputs.
            throw std::runtime_error("Operation with no tensor inputs not yet supported.");
        }

        // Determine target device for execution
        core::DeviceType target_device =
            inputs_for_op[0]->device(); // Simple heuristic: use first input's device
        for (size_t i = 1; i < inputs_for_op.size(); ++i)
        {
            if (inputs_for_op[i]->device() != target_device)
            {
                // For now, throw error if inputs are on different devices.
                // Later, implement automatic device transfer or more sophisticated device
                // placement.
                throw std::runtime_error("Inputs to an operation are on different devices. "
                                         "Automatic transfer not yet implemented.");
            }
        }

        if (target_device == core::DeviceType::CPU)
        {
            tensor::Tensor output_tensor = node->operation()->execute_cpu(inputs_for_op);
            node->set_cached_value(std::move(output_tensor)); // Cache the result
        }
        else if (target_device == core::DeviceType::CUDA)
        {
            tensor::Tensor output_tensor = node->operation()->execute_cuda(inputs_for_op);
            node->set_cached_value(std::move(output_tensor)); // Cache the result
        }
        else
        {
            throw std::runtime_error("Unsupported device type for operation execution.");
        }
    }

    // The result of the root node is the final output
    if (!root_node->has_cached_value())
    {
        throw std::runtime_error("Root node value not computed after graph execution.");
    }
    return root_node->get_cached_value().clone();
}

void ExecutionEngine::clear_cache()
{
    // This would ideally traverse the graph and clear all cached values.
    // For now, we rely on the fact that `execute` will recompute everything
    // and overwrite any old cached values. A more robust solution would
    // involve iterating through all nodes reachable from the root and clearing their caches.
    // This is a placeholder for a more complete implementation.
    // For now, the `execute` method implicitly clears by recomputing.
}

} // namespace execution
} // namespace plast
