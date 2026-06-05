#include <stdlib.h>
#include "scheduler/scheduler.h"

Scheduler *init_scheduler(JIT *jit) {
  Scheduler *scheduler = malloc(sizeof(Scheduler));
  scheduler->jit = jit;
  scheduler->jit_mode = false;
  return scheduler;
}

void set_jit_mode(Scheduler *scheduler, bool jit_mode) {
  scheduler->jit_mode = jit_mode;
}

void schedule(Scheduler *scheduler, Node *root, PASS pass) {
  DAG *dag = alloc_dag(MIN_DAG_CAPACITY);
  build_dag(dag, root);

  bool fresh = true;

  if (scheduler->jit_mode) {
    u32 slot = search(scheduler->jit, dag);
    if (slot == (u32)-1) {
      cache(scheduler->jit, dag);
      fresh = false;
    }
  }

  if (pass == FORWARD)
    dag_forward(dag);
  else
    dag_backward(dag);

  if (fresh)
    dag_release(dag);
}

void scheduler_release(Scheduler *scheduler) {
  jit_release(scheduler->jit);
  free(scheduler);
}
