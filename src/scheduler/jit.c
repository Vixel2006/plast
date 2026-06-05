#include <stdlib.h>
#include <string.h>
#include "scheduler/jit.h"

JIT *init_jit(u32 cap) {
  JIT *jit = malloc(sizeof(JIT));

  jit->count = 0;
  jit->cap = cap;
  jit->cached_jobs = calloc(cap, sizeof(DAG *));

  return jit;
}

// NOTE: Here we use the FNV-1a hashing function with unsigned 32-bit integers
u32 hash(JIT *jit, DAG *dag) {
  u32 fnv_hash = 2166136261u;
  u32 fnv_offset_basis = 16777619u;

  fnv_hash ^= dag->count;
  fnv_hash *= fnv_offset_basis;

  for (u32 i = 0; i < dag->count; ++i) {
    Node *n = dag->nodes[i];

    fnv_hash ^= (u32)n->op_type;
    fnv_hash *= fnv_offset_basis;

    fnv_hash ^= (u32)n->num_inputs;
    fnv_hash *= fnv_offset_basis;

    // here we do this pattern as param.dim is u64
    // so we basically do this sliding window logic for completing the hasing over the full value
    fnv_hash ^= (u32)(n->params.dim & 0xFFFFFFFF);
    fnv_hash *= fnv_offset_basis;
    fnv_hash ^= (u32)(n->params.dim >> 32);
    fnv_hash *= fnv_offset_basis;

    fnv_hash ^= (u32)(n->params.keepdim & 0xFFFFFFFF);
    fnv_hash *= fnv_offset_basis;

    fnv_hash ^= (u32)(n->params.keepdim >> 32);
    fnv_hash *= fnv_offset_basis;

    u32 fv;
    memcpy(&fv, &n->params.fval, sizeof(fv));
    fnv_hash ^= fv;
    fnv_hash *= fnv_offset_basis;

    if (n->output) {
      fnv_hash ^= (u32)n->output->ndim;
      fnv_hash *= fnv_offset_basis;

      fnv_hash ^= (u32)n->output->dtype;
      fnv_hash *= fnv_offset_basis;

      for (u32 d = 0; d < n->output->ndim; ++d) {
        fnv_hash ^= (u32)(n->output->shape[d] & 0xFFFFFFFF);
        fnv_hash *= fnv_offset_basis;

        fnv_hash ^= (u32)(n->output->shape[d] >> 32);
        fnv_hash *= fnv_offset_basis;
      }
    }
  }

  return fnv_hash % jit->cap;
}

u32 search(JIT *jit, DAG *dag) {
  u32 idx = hash(jit, dag);
  u32 st_slot = idx;

  while (jit->cached_jobs[idx] != NULL) {
    if (dag_equal(jit->cached_jobs[idx], dag))
      return idx;

    idx = (idx + 1) % jit->cap;

    if (idx == st_slot)
      break;
  }

  return -1;
}

void cache(JIT *jit, DAG *dag) {
  // if the slot returns any value other than -1 then our dag is cached.
  // this way we can tell the scheduler to run the graph directly without running optimization runs
  // if it returns -1 this means that it is not compiled before.
  // so this way we will make the scheduler do some pattern matching on it for optimization.
  u32 slot = search(jit, dag);

  if (slot != -1) {
    return;
  }

  // Here we multiple the cap by 0.7 so that we resize if the load factor > 70%
  // this way we can reduce collision problems
  if (jit->count >= jit->cap * 0.7) {
    DAG **old_jobs = jit->cached_jobs;
    u32 old_cap = jit->cap;

    jit->cap *= 2;
    jit->count = 0;
    jit->cached_jobs = calloc(jit->cap, sizeof(DAG *));

    for (u32 i = 0; i < old_cap; ++i) {
      if (old_jobs[i] != NULL) {
        cache(jit, old_jobs[i]);
      }
    }

    free(old_jobs);
  }

  // we will use linear proping to resolve collisions
  u32 idx = hash(jit, dag);
  while (jit->cached_jobs[idx] != NULL) {
    if (jit->cached_jobs[idx] == dag) return; // already in table
    idx = (idx + 1) % jit->cap;
  }

  jit->cached_jobs[idx] = dag;
  jit->count++;
}

void jit_release(JIT *jit) {
  if (jit->cached_jobs) {
    for (int i = 0; i < jit->count; ++i)
      dag_release(jit->cached_jobs[i]);
  }
  free(jit);
}
