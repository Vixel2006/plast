#include "graph.h"

#define MIN_CAPACITY 1

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
  if (!node || !node->visited) return;
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

void forward(Node *node) {
  DAG *dag = alloc_dag(MIN_CAPACITY);
  build_dag(dag, node);

  for (u64 i = 0; i < dag->count; ++i) {
    execute_forward(dag->nodes[i]);
  }
  dag_release(dag);
}

void backward(Node *node) {
  DAG *dag = alloc_dag(MIN_CAPACITY);
  build_dag(dag, node);

  for (u64 i = dag->count; i > 0; --i) {
    execute_backward(dag->nodes[i - 1]);
  }
  dag_release(dag);
}

void dag_release(DAG *dag) {
  free(dag->nodes);
  free(dag);
}
