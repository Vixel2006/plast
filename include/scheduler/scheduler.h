#ifndef SCHEDULER_H
#define SCHEDULER_H

#include "core/definitions.h"
#include "core/graph.h"
#include "scheduler/jit.h"

typedef enum { FORWARD, BACKWARD } PASS;

typedef struct Scheduler {
  JIT *jit;
  bool jit_mode;

} Scheduler;

Scheduler *init_scheduler(JIT *jit);
void schedule(Scheduler *scheduler, Node *root, PASS pass);
void set_jit_mode(Scheduler *scheduler, bool jit_mode);
void scheduler_release(Scheduler *scheduler);

#endif // SCHEDULER_H
