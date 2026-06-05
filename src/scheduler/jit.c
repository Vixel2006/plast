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

u32 cache(JIT *jit, DAG *dag) {
  // if the slot returns any value other than -1 then our dag is cached.
  // this way we can tell the scheduler to run the graph directly without running optimization runs
  // if it returns -1 this means that it is not compiled before.
  // so this way we will make the scheduler do some pattern matching on it for optimization.
  u32 slot = search(jit, dag);

  if (slot != -1) {
    return -1;
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
    if (jit->cached_jobs[idx] == dag)
      return idx;
    idx = (idx + 1) % jit->cap;
  }

  jit->cached_jobs[idx] = dag;
  jit->count++;

  return idx;
}

void jit_release(JIT *jit) {
  for (u32 i = 0; i < jit->cap; ++i) {
    if (jit->cached_jobs[i] != NULL)
      dag_release(jit->cached_jobs[i]);
  }
  free(jit->cached_jobs);
  free(jit);
}

#if defined(JIT_TEST)
#include <stdio.h>

static Node *test_node(OP_TYPE op_type, int num_inputs, u64 dim, u64 keepdim, float fval,
                       u64 *shape, u64 ndim) {
  Node *n = calloc(1, sizeof(Node));
  n->op_type = op_type;
  n->num_inputs = num_inputs;
  n->params.dim = dim;
  n->params.keepdim = keepdim;
  n->params.fval = fval;

  Tensor *t = calloc(1, sizeof(Tensor));
  t->ndim = ndim;
  t->dtype = FLOAT32;
  t->device = CPU;
  for (u64 d = 0; d < ndim; d++)
    t->shape[d] = shape[d];
  n->output = t;

  return n;
}

static DAG *test_dag(Node **nodes, u32 count) {
  DAG *dag = malloc(sizeof(DAG));
  dag->nodes = malloc(sizeof(Node *) * count);
  for (u32 i = 0; i < count; i++)
    dag->nodes[i] = nodes[i];
  dag->count = count;
  dag->capacity = count;
  dag->changed = false;
  return dag;
}

void dag_release(DAG *dag) {
  for (u32 i = 0; i < dag->count; i++) {
    if (dag->nodes[i]->output)
      free(dag->nodes[i]->output);
    free(dag->nodes[i]);
  }
  free(dag->nodes);
  free(dag);
}

bool dag_equal(DAG *a, DAG *b) {
  if (a->count != b->count)
    return false;
  for (u32 i = 0; i < a->count; i++) {
    Node *na = a->nodes[i];
    Node *nb = b->nodes[i];
    if (na->op_type != nb->op_type)
      return false;
    if (na->num_inputs != nb->num_inputs)
      return false;
    if (na->params.dim != nb->params.dim)
      return false;
    if (na->params.keepdim != nb->params.keepdim)
      return false;
    if (na->params.fval != nb->params.fval)
      return false;
    if ((na->output == NULL) != (nb->output == NULL))
      return false;
    if (na->output) {
      if (na->output->ndim != nb->output->ndim)
        return false;
      if (na->output->dtype != nb->output->dtype)
        return false;
      if (na->output->device != nb->output->device)
        return false;
      for (u64 d = 0; d < na->output->ndim; d++) {
        if (na->output->shape[d] != nb->output->shape[d])
          return false;
      }
    }
  }
  return true;
}

int main() {
  u32 pass = 0, fail = 0;
  JIT *jit = init_jit(4);

  // Test 1: same DAG → same hash
  u64 shape_4x4[] = {4, 4};
  Node *n1_a = test_node(ADD, 2, 0, 0, 0.0f, shape_4x4, 2);
  Node *n2_a = test_node(MUL, 2, 0, 0, 0.0f, shape_4x4, 2);
  Node *arr_a[] = {n1_a, n2_a};
  DAG *dag_a = test_dag(arr_a, 2);

  Node *n1_b = test_node(ADD, 2, 0, 0, 0.0f, shape_4x4, 2);
  Node *n2_b = test_node(MUL, 2, 0, 0, 0.0f, shape_4x4, 2);
  Node *arr_b[] = {n1_b, n2_b};
  DAG *dag_b = test_dag(arr_b, 2);

  u32 h_a = hash(jit, dag_a);
  u32 h_b = hash(jit, dag_b);
  if (h_a == h_b) {
    printf("PASS");
    pass++;
  } else {
    printf("FAIL");
    fail++;
  }
  printf(": same structure → hash %u == %u\n", h_a, h_b);

  // Test 2: search empty table → -1
  u32 idx = search(jit, dag_a);
  if (idx == (u32)-1) {
    printf("PASS");
    pass++;
  } else {
    printf("FAIL");
    fail++;
  }
  printf(": search before cache → %u\n", idx);

  // Test 3: cache + search
  cache(jit, dag_a);
  idx = search(jit, dag_a);
  if (idx != (u32)-1) {
    printf("PASS");
    pass++;
  } else {
    printf("FAIL");
    fail++;
  }
  printf(": cache then same DAG → idx=%u\n", idx);

  // Test 4: search structurally identical DAG → found
  idx = search(jit, dag_b);
  if (idx != (u32)-1) {
    printf("PASS");
    pass++;
  } else {
    printf("FAIL");
    fail++;
  }
  printf(": search equivalent DAG → idx=%u\n", idx);

  // Test 5: different DAG (diff op_type) → not found
  Node *n3 = test_node(SIN, 1, 0, 0, 0.0f, shape_4x4, 2);
  DAG *dag_c = test_dag(&n3, 1);
  idx = search(jit, dag_c);
  if (idx == (u32)-1) {
    printf("PASS");
    pass++;
  } else {
    printf("FAIL");
    fail++;
  }
  printf(": different DAG (SIN vs ADD/MUL) → not found\n");

  // Test 6: different fval → dag_equal says not equal
  Node *n4_a = test_node(LEAKY_RELU, 1, 0, 0, 0.01f, shape_4x4, 2);
  DAG *dag_d = test_dag(&n4_a, 1);
  Node *n4_b = test_node(LEAKY_RELU, 1, 0, 0, 0.10f, shape_4x4, 2);
  DAG *dag_e = test_dag(&n4_b, 1);
  cache(jit, dag_d);
  idx = search(jit, dag_e);
  if (idx == (u32)-1) {
    printf("PASS");
    pass++;
  } else {
    printf("FAIL");
    fail++;
  }
  printf(": diff fval (alpha 0.01 vs 0.1) → not found via dag_equal\n");

  // Test 7: different shape → dag_equal says not equal (even if hash collides)
  u64 shape_8x8[] = {8, 8};
  Node *n5 = test_node(ADD, 2, 0, 0, 0.0f, shape_8x8, 2);
  DAG *dag_f = test_dag(&n5, 1);
  Node *n6 = test_node(ADD, 2, 0, 0, 0.0f, shape_4x4, 2);
  DAG *dag_g = test_dag(&n6, 1);
  cache(jit, dag_f);
  idx = search(jit, dag_g);
  if (idx == (u32)-1) {
    printf("PASS");
    pass++;
  } else {
    printf("FAIL");
    fail++;
  }
  printf(": diff shape (8x8 vs 4x4) → not found via dag_equal\n");

  // Test 8: cache many DAGs past resize threshold
  DAG *dags[20];
  for (int i = 0; i < 20; i++) {
    u64 s[] = {4, (u64)(i + 1)};
    Node *node = test_node(MUL, 2, i, 0, (float)i, s, 2);
    dags[i] = test_dag(&node, 1);
    cache(jit, dags[i]);
  }
  int all_found = 1;
  for (int i = 0; i < 20; i++) {
    if (search(jit, dags[i]) == (u32)-1) {
      all_found = 0;
      break;
    }
  }
  if (all_found) {
    printf("PASS");
    pass++;
  } else {
    printf("FAIL");
    fail++;
  }
  printf(": %d DAGs after resize → all findable (cap=%u, count=%u)\n", 20, jit->cap, jit->count);

  printf("\n--- Results: %u/%u passed ---\n", pass, pass + fail);

  // Cleanup uncached DAGs
  dag_release(dag_b);
  dag_release(dag_c);
  dag_release(dag_e);
  dag_release(dag_g);

  jit_release(jit);

  return fail > 0 ? 1 : 0;
}
#endif
