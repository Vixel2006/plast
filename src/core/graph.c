#include "core/graph.h"

DAG *alloc_dag(u32 capacity) {
  DAG *dag = malloc(sizeof(DAG));

  dag->capacity = capacity;
  dag->nodes = malloc(sizeof(Node *) * capacity);
  dag->count = 0;
  dag->changed = false;

  return dag;
}

void insert_node(DAG *dag, Node *node) {
  if (dag->count >= dag->capacity) {
    dag->capacity *= 2;
    Node **nodes = malloc(sizeof(Node *) * dag->capacity);

    for (u64 i = 0; i < dag->count; ++i) {
      nodes[i] = dag->nodes[i];
    }

    free(dag->nodes);
    dag->nodes = nodes;
  }
  dag->nodes[dag->count++] = node;
}

void topological_sort(DAG *dag, Node *root) {
  root->visited = true;

  for (u64 i = 0; i < root->num_inputs; ++i) {
    if (root->inputs[i]->creator && !root->inputs[i]->creator->visited) {
      topological_sort(dag, root->inputs[i]->creator);
    }
  }
  insert_node(dag, root);
}

void reset_node_flags(Node *node) {
  if (!node || !node->visited)
    return;
  node->visited = false;
  for (u64 i = 0; i < node->num_inputs; ++i) {
    if (node->inputs[i]->creator) {
      reset_node_flags(node->inputs[i]->creator);
    }
  }
}

void build_dag(DAG *dag, Node *root) {
  reset_node_flags(root);
  topological_sort(dag, root);
}

void dag_forward(DAG *dag) {
  for (u32 i = 0; i < dag->count; ++i)
    execute_forward(dag->nodes[i]);
}

void dag_backward(DAG *dag) {
  for (u32 i = dag->count; i > 0; --i)
    execute_backward(dag->nodes[i - 1]);
}

void forward(Node *node) {
  DAG *dag = alloc_dag(MIN_DAG_CAPACITY);
  build_dag(dag, node);
  dag_forward(dag);
  dag_release(dag);
}

void backward(Node *node) {
  DAG *dag = alloc_dag(MIN_DAG_CAPACITY);
  build_dag(dag, node);
  dag_backward(dag);
  dag_release(dag);
}

bool dag_equal(DAG *lhs, DAG *rhs) {
  if (lhs->count != rhs->count) return false;
  for (u32 i = 0; i < lhs->count; i++) {
    Node *a = lhs->nodes[i];
    Node *b = rhs->nodes[i];
    if (a->op_type != b->op_type) return false;
    if (a->num_inputs != b->num_inputs) return false;
    if (a->params.dim != b->params.dim) return false;
    if (a->params.keepdim != b->params.keepdim) return false;
    if (a->params.fval != b->params.fval) return false;
    if ((a->output == NULL) != (b->output == NULL)) return false;
    if (a->output) {
      if (a->output->ndim != b->output->ndim) return false;
      if (a->output->dtype != b->output->dtype) return false;
      if (a->output->device != b->output->device) return false;
      for (u64 d = 0; d < a->output->ndim; d++) {
        if (a->output->shape[d] != b->output->shape[d]) return false;
      }
    }
  }
  return true;
}

void dag_release(DAG *dag) {
  free(dag->nodes);
  free(dag);
}
