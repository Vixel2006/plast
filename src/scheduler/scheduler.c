#include <stdlib.h>
#include "scheduler/scheduler.h"

Scheduler *init_scheduler(JIT *jit) {
  Scheduler *scheduler = malloc(sizeof(Scheduler));
  scheduler->jit = jit;
  return scheduler;
}

void schedule(Scheduler *scheduler, Node *root, PASS pass) {
  DAG *dag = alloc_dag(MIN_DAG_CAPACITY);
  build_dag(dag, root);

  u32 slot = search(scheduler->jit, dag);
  if (slot == (u32)-1) {
    slot = cache(scheduler->jit, dag);
  } else {
    dag_release(dag);
    dag = scheduler->jit->cached_jobs[slot];
  }

  // TODO: Here we should add the pattern matching for optimization.

  if (pass == FORWARD)
    dag_forward(dag);
  else
    dag_backward(dag);
}

void scheduler_release(Scheduler *scheduler) {
  jit_release(scheduler->jit);
  free(scheduler);
}
