#ifndef SCHEDULER_H
#define SCHEDULER_H

#include "core/definitions.h"
#include "core/graph.h"
#include "scheduler/jit.h"

typedef enum { FORWARD, BACKWARD } PASS;

typedef struct Scheduler {
  JIT *jit;

  // TODO: the pattern matcher artifacts should be added here.

} Scheduler;

Scheduler *init_scheduler(JIT *jit);
void schedule(Scheduler *scheduler, Node *root, PASS pass);
void scheduler_release(Scheduler *scheduler);

#endif // SCHEDULER_H
