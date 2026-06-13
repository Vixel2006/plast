#ifndef JIT_H
#define JIT_H

#include "core/definitions.h"
#include "core/node.h"
#include "core/graph.h"

#define JIT_LOAD_FACTOR 0.7f

typedef struct JITEntry {
  u64 fingerprint;
  DAG *dag;
  bool occupied;
} JITEntry;

typedef struct JIT {
  JITEntry *entries;
  u32 cap;
  u32 count;
} JIT;

JIT *jit_create(u32 cap);
u64 jit_fingerprint(Node *root);
DAG *jit_lookup(JIT *jit, u64 fp);
bool jit_insert(JIT *jit, u64 fp, DAG *dag);
void jit_clear(JIT *jit);
void jit_release(JIT *jit);

#endif // JIT_H
