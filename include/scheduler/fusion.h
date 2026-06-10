#ifndef FUSION_H
#define FUSION_H

#include "core/definitions.h"
#include "core/graph.h"

typedef enum {
  FUSION_NONE = 0,
  FUSION_MATMUL_RELU,
  FUSION_MATMUL_BIAS_RELU,
  FUSION_CONV_RELU,
} FusionPatternType;

typedef struct {
  FusionPatternType type;
  Node **nodes;
  u32 num_nodes;
  bool valid;
} FusionMatch;

FusionMatch *alloc_pattern(Node **nodes, u32 num_nodes, FusionPatternType type);
void pattern_release(FusionMatch *pattern);

u32 fusion_find_patterns(DAG *dag, FusionMatch *matches, u32 max_matches);
bool fusion_apply(DAG *dag, FusionMatch *match);
u32 fusion_optimize(DAG *dag);

#endif // FUSION_H
