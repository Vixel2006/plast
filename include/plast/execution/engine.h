#pragma once

#include <memory>
#include <unordered_map>

#include "plast/graph/node.h"
#include "plast/tensor/tensor.h"

namespace plast
{
namespace execution
{

class ExecutionEngine
{
  public:
    ExecutionEngine();
    ~ExecutionEngine();

    // Executes the computation graph rooted at 'root_node'
    std::shared_ptr<tensor::Tensor> execute(std::shared_ptr<graph::Node> root_node);

  private:
    // Internal helper to perform topological sort
    std::vector<std::shared_ptr<graph::Node>>
    topological_sort(std::shared_ptr<graph::Node> root_node);

    // Internal helper to recursively visit nodes for topological sort
    void visit(std::shared_ptr<graph::Node> node,
               std::unordered_map<std::shared_ptr<graph::Node>, bool>& visited,
               std::unordered_map<std::shared_ptr<graph::Node>, bool>& in_stack,
               std::vector<std::shared_ptr<graph::Node>>& sorted_nodes);
};

} // namespace execution
} // namespace plast
