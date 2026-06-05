#ifndef JIT_H
#define JIT_H

#include "core/definitions.h"
#include "core/arena.h"
#include "core/graph.h"

typedef struct JIT {
  DAG **cached_jobs;
  u32 count;
  u32 cap;
} JIT;

JIT *init_jit(u32 cap);
u32 hash(JIT *jit, DAG *dag);
u32 search(JIT *jit, DAG *dag);
void cache(JIT *jit, DAG *dag);
void jit_release(JIT *jit);

#endif // JIT_H
