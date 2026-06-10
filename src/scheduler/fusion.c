#include "scheduler/fusion.h"
#include "core/op.h"
#include <stdlib.h>
#include <string.h>

FusionMatch *alloc_pattern(Node **nodes, u32 num_nodes, FusionPatternType type) {
  FusionMatch *pattern = malloc(sizeof(FusionMatch));
  pattern->type = type;
  pattern->valid = true;
  pattern->num_nodes = num_nodes;
  pattern->nodes = malloc(sizeof(Node *) * num_nodes);
  for (u32 i = 0; i < num_nodes; ++i)
    pattern->nodes[i] = nodes[i];
  return pattern;
}

void pattern_release(FusionMatch *pattern) {
  if (pattern) {
    free(pattern->nodes);
    free(pattern);
  }
}

static bool is_bias_like(const Tensor *t) {
  return t->ndim == 1;
}

static u32 try_match_ending_at(DAG *dag, u32 node_idx, FusionMatch *match) {
  Node *n = dag->nodes[node_idx];

  switch (n->op_type) {

  case LEAKY_RELU: {
    if (n->num_inputs != 1) return 0;
    Node *producer = n->inputs[0]->creator;
    if (!producer) return 0;

    if (producer->op_type == MATMUL) {
      // MATMUL -> LEAKY_RELU
      match->type = FUSION_MATMUL_RELU;
      match->nodes = malloc(2 * sizeof(Node *));
      match->nodes[0] = producer;
      match->nodes[1] = n;
      match->num_nodes = 2;
      match->valid = true;
      return 1;
    }

    if (producer->op_type == ADD) {
      // Check ADD's first input comes from MATMUL (bias case)
      Node *add = producer;
      if (add->num_inputs < 2) return 0;
      Node *add_lhs = add->inputs[0]->creator;
      if (!add_lhs || add_lhs->op_type != MATMUL) return 0;
      if (!is_bias_like(add->inputs[1])) return 0;

      // MATMUL -> ADD( bias ) -> LEAKY_RELU
      match->type = FUSION_MATMUL_BIAS_RELU;
      match->nodes = malloc(3 * sizeof(Node *));
      match->nodes[0] = add_lhs;    // MATMUL
      match->nodes[1] = add;        // ADD
      match->nodes[2] = n;          // LEAKY_RELU
      match->num_nodes = 3;
      match->valid = true;
      return 1;
    }

    if (producer->op_type == CONV2D) {
      // CONV2D -> LEAKY_RELU
      match->type = FUSION_CONV_RELU;
      match->nodes = malloc(2 * sizeof(Node *));
      match->nodes[0] = producer;
      match->nodes[1] = n;
      match->num_nodes = 2;
      match->valid = true;
      return 1;
    }

    return 0;
  }

  default:
    return 0;
  }
}

u32 fusion_find_patterns(DAG *dag, FusionMatch *matches, u32 max_matches) {
  u32 found = 0;

  for (u32 i = 0; i < dag->count && found < max_matches; ++i) {
    Node *n = dag->nodes[i];
    if (n->op_type == LEAKY_RELU) {
      found += try_match_ending_at(dag, i, &matches[found]);
    }
  }

  return found;
}

bool fusion_apply(DAG *dag, FusionMatch *match) {
  Node *fused = malloc(sizeof(Node));
  memset(fused, 0, sizeof(Node));

  switch (match->type) {

  case FUSION_MATMUL_RELU: {
    Node *mm = match->nodes[0];
    Node *relu = match->nodes[1];
    fused->op_type = MATMUL_RELU;
    fused->op = get_op_impl(MATMUL_RELU);
    fused->inputs = mm->inputs;
    fused->num_inputs = mm->num_inputs;
    fused->output = relu->output;
    fused->params = relu->params;  // preserves fval (alpha)
    break;
  }

  case FUSION_MATMUL_BIAS_RELU: {
    Node *mm = match->nodes[0];
    Node *add = match->nodes[1];
    Node *relu = match->nodes[2];
    fused->op_type = MATMUL_BIAS_RELU;
    fused->op = get_op_impl(MATMUL_BIAS_RELU);
    fused->inputs = malloc(3 * sizeof(Tensor *));
    fused->inputs[0] = mm->inputs[0];
    fused->inputs[1] = mm->inputs[1];
    fused->inputs[2] = add->inputs[1];
    fused->num_inputs = 3;
    fused->output = relu->output;
    fused->params = relu->params;
    break;
  }

  case FUSION_CONV_RELU: {
    Node *conv = match->nodes[0];
    Node *relu = match->nodes[1];
    fused->op_type = CONV_RELU;
    fused->op = get_op_impl(CONV_RELU);
    fused->inputs = conv->inputs;
    fused->num_inputs = conv->num_inputs;
    fused->output = relu->output;
    fused->params = relu->params;
    break;
  }

  default:
    free(fused);
    return false;
  }

  // Wire the output tensor's creator to the fused node
  if (fused->output)
    fused->output->creator = fused;

  // Find indices of matched nodes in the DAG array
  u32 count = match->num_nodes;
  u32 *indices = calloc(count, sizeof(u32));
  u32 found = 0;
  for (u32 i = 0; i < dag->count && found < count; ++i) {
    for (u32 j = 0; j < count; ++j) {
      if (dag->nodes[i] == match->nodes[j]) {
        indices[found++] = i;
        break;
      }
    }
  }

  if (found != count) {
    free(indices);
    free(fused);
    return false;
  }

  // Sort indices ascending
  for (u32 i = 0; i < count; ++i) {
    for (u32 j = i + 1; j < count; ++j) {
      if (indices[j] < indices[i]) {
        u32 t = indices[i]; indices[i] = indices[j]; indices[j] = t;
      }
    }
  }

  // Replace the first matched slot with the fused node
  dag->nodes[indices[0]] = fused;

  // Compact: remove other matched nodes by shifting left
  u32 write = indices[0] + 1;
  u32 skip = 1;
  for (u32 read = indices[0] + 1; read < dag->count; ++read) {
    if (skip < count && read == indices[skip]) {
      skip++;
      continue;
    }
    dag->nodes[write++] = dag->nodes[read];
  }

  dag->count -= (count - 1);
  dag->changed = true;

  free(indices);
  return true;
}

u32 fusion_optimize(DAG *dag) {
  // Upper bound: each match consumes 2+ nodes, DAG has at most count/2 matches.
  u32 max_matches = dag->count / 2;
  FusionMatch *matches = malloc(max_matches * sizeof(FusionMatch));
  u32 num = fusion_find_patterns(dag, matches, max_matches);

  u32 applied = 0;
  for (u32 i = 0; i < num; ++i) {
    if (fusion_apply(dag, &matches[i]))
      applied++;
    free(matches[i].nodes);
  }

  free(matches);
  return applied;
}
