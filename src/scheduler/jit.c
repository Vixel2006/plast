#include <stdlib.h>
#include <string.h>
#include "scheduler/jit.h"
#include "core/graph.h"

#define FNV_OFFSET_64 14695981039346656037ULL
#define FNV_PRIME_64 1099511628211ULL

static void hash_byte(u64 *h, u8 byte) {
  *h ^= byte;
  *h *= FNV_PRIME_64;
}

static void hash_mem(u64 *h, const void *data, u32 size) {
  const u8 *bytes = (const u8 *)data;
  for (u32 i = 0; i < size; ++i)
    hash_byte(h, bytes[i]);
}

static void hash_u32(u64 *h, u32 v) {
  hash_mem(h, &v, sizeof(v));
}
static void hash_u64(u64 *h, u64 v) {
  hash_mem(h, &v, sizeof(v));
}
static void hash_f32(u64 *h, float v) {
  hash_mem(h, &v, sizeof(v));
}

static void fingerprint_node(Node *node, u64 *h) {
  if (!node || node->visited)
    return;
  node->visited = true;

  for (u64 i = 0; i < node->num_inputs; ++i) {
    if (node->inputs[i]->creator)
      fingerprint_node(node->inputs[i]->creator, h);
  }

  hash_u32(h, (u32)node->op_type);
  hash_u32(h, (u32)node->num_inputs);
  hash_u64(h, node->params.dim);
  hash_u64(h, node->params.keepdim);
  hash_f32(h, node->params.fval);

  if (node->output) {
    hash_u64(h, node->output->ndim);
    hash_u32(h, (u32)node->output->dtype);
    for (u64 d = 0; d < node->output->ndim; ++d)
      hash_u64(h, node->output->shape[d]);
  }
}

u64 jit_fingerprint(Node *root) {
  u64 h = FNV_OFFSET_64;
  reset_node_flags(root);
  fingerprint_node(root, &h);
  reset_node_flags(root);
  return h;
}

JIT *jit_create(u32 cap) {
  JIT *jit = malloc(sizeof(JIT));
  jit->cap = cap ? cap : 64;
  jit->count = 0;
  jit->entries = calloc(jit->cap, sizeof(JITEntry));
  return jit;
}

DAG *jit_lookup(JIT *jit, u64 fp) {
  u32 idx = (u32)(fp % jit->cap);
  for (u32 i = 0; i < jit->cap; ++i) {
    u32 slot = (idx + i) % jit->cap;
    if (!jit->entries[slot].occupied)
      return NULL;
    if (jit->entries[slot].fingerprint == fp)
      return jit->entries[slot].dag;
  }
  return NULL;
}

static void jit_resize(JIT *jit) {
  JITEntry *old = jit->entries;
  u32 old_cap = jit->cap;
  jit->cap *= 2;
  jit->count = 0;
  jit->entries = calloc(jit->cap, sizeof(JITEntry));
  for (u32 i = 0; i < old_cap; ++i) {
    if (old[i].occupied)
      jit_insert(jit, old[i].fingerprint, old[i].dag);
  }
  free(old);
}

bool jit_insert(JIT *jit, u64 fp, DAG *dag) {
  if (jit_lookup(jit, fp))
    return false;

  if ((float)jit->count / (float)jit->cap >= JIT_LOAD_FACTOR)
    jit_resize(jit);

  u32 idx = (u32)(fp % jit->cap);
  for (u32 i = 0; i < jit->cap; ++i) {
    u32 slot = (idx + i) % jit->cap;
    if (!jit->entries[slot].occupied) {
      jit->entries[slot].fingerprint = fp;
      jit->entries[slot].dag = dag;
      jit->entries[slot].occupied = true;
      jit->count++;
      return true;
    }
  }
  return false;
}

void jit_clear(JIT *jit) {
  for (u32 i = 0; i < jit->cap; ++i) {
    if (jit->entries[i].occupied) {
      dag_release(jit->entries[i].dag);
      jit->entries[i].occupied = false;
      jit->entries[i].dag = NULL;
    }
  }
  jit->count = 0;
}

void jit_release(JIT *jit) {
  jit_clear(jit);
  free(jit->entries);
  free(jit);
}
