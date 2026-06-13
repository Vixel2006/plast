#include <stdlib.h>
#include "scheduler/scheduler.h"
#include "scheduler/fusion.h"

Scheduler *init_scheduler(JIT *jit) {
  Scheduler *scheduler = malloc(sizeof(Scheduler));
  scheduler->jit = jit;
  scheduler->jit_mode = false;
  return scheduler;
}

void set_jit_mode(Scheduler *scheduler, bool jit_mode) {
  scheduler->jit_mode = jit_mode;
}

void schedule(Scheduler *scheduler, Node *root, PASS pass, Arena *arena) {
  if (scheduler->jit_mode) {
    u64 fp = jit_fingerprint(root);
    DAG *cached = jit_lookup(scheduler->jit, fp);

    if (cached) {
      if (pass == FORWARD)
        dag_forward(cached);
      else
        dag_backward(cached);
      return;
    }

    DAG *dag = alloc_dag(MIN_DAG_CAPACITY);
    dag->arena = arena;
    build_dag(dag, root);
    fusion_optimize(dag);
    jit_insert(scheduler->jit, fp, dag);

    if (pass == FORWARD)
      dag_forward(dag);
    else
      dag_backward(dag);
    return;
  }

  DAG *dag = alloc_dag(MIN_DAG_CAPACITY);
  dag->arena = arena;
  build_dag(dag, root);

  if (pass == FORWARD)
    dag_forward(dag);
  else
    dag_backward(dag);

  dag_release(dag);
}

void scheduler_release(Scheduler *scheduler) {
  jit_release(scheduler->jit);
  free(scheduler);
}
