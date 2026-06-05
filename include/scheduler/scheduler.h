#ifndef SCHEDULER_H
#define SCHEDULER_H

#include "core/definitions.h"

/*
 * Some shit to consider. I think here the scheduler will be like a pattern matcher. or something. so it will take the dag.
 * and it will dot he full optimization on it fusing kernels mainly as I don't have plans for more shit right now.
 * so basically maybe the JIT will be the struct that will take the graph hash. if the hash as not there it will go to the scheduler.
 * if there is a hash. then it will implement the known graph directly
 * this means that we will need a hash function and a hash table.
 * this way the JIT compiler will have a table of graphs. each graph we get we will see the hash of it. if there. just take the graph from the jit and implement.
 * if not in there we can basically go to the scheudler. that will do the pattern matching on this graph
 * */

#endif // SCHEDULER_H
