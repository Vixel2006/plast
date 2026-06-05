#pragma once

#include "core/node.h"

typedef struct DAG {
  Node **nodes;
  u32 count;
  u32 capacity;
  bool changed;
} DAG;

DAG *alloc_dag(u32 capacity);
void insert_node(DAG *dag, Node *node);
void topological_sort(DAG *dag, Node *root);
void build_dag(DAG *dag, Node *root);
void forward(Node *node);
void backward(Node *node);
bool dag_equal(DAG *lhs, DAG *rhs);
void dag_release(DAG *dag);
void reset_node_flags(Node *node);
