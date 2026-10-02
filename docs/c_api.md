# C API Guide

Plast is implemented as a lightweight C library. The Python bindings are a thin wrapper over this core. This document explains how to write standalone C applications, use the memory arenas, construct graphs, and compile/link your projects.

---

## 1. Core Header Files

To build Plast applications in C, include the following headers in your files:

```c
#include "core/arena.h"       // Arena allocations and device pools
#include "core/tensor.h"      // Tensor structures and creation
#include "core/node.h"        // DAG nodes and kernel hyper-parameters
#include "core/op.h"          // OP_TYPE registry and implementations
#include "scheduler/scheduler.h" // DAG execution schedules and graph JITs
#include "optimizers/sgd.h"   // Native parameters stepping (e.g. SGD/Adam)
```

---

## 2. Memory Allocations

Plast uses manual memory pools. Dynamically allocated tensors must have their metadata and data tracked inside active pools.

### Creating Arenas
```c
// Create a CPU arena for graph metadata (nodes, topological trackers, etc.)
Arena meta_arena = arena_create(Mib(10), CPU);

// Create a GPU/CUDA arena for actual tensor data buffers
Arena data_arena = arena_create(Mib(100), CUDA);
```

### Initializing Tensors
You initialize tensors inside active arenas using the `init` function. You can optionally specify a custom initialization function (e.g., `rand_init` or `zeros`):

```c
// Allocates an input tensor of shape [4, 2] on CUDA without gradient tracking
Tensor *X = init(&meta_arena, &data_arena, CUDA, FLOAT32, (u64[]){4, 2}, 2, false, NULL);

// Allocates a parameter weight tensor of shape [2, 8] with random initialization
Tensor *W = init(&meta_arena, &data_arena, CUDA, FLOAT32, (u64[]){2, 8}, 2, true, rand_init);
```

---

## 3. Constructing Computation Nodes

Operations are represented in the computation graph as `Node` structs. You allocate nodes directly inside your metadata arena:

```c
// Create output buffer to house matrix multiplication outputs [4, 8]
Tensor *out = init(&meta_arena, &data_arena, CUDA, FLOAT32, (u64[]){4, 8}, 2, true, NULL);

// Allocate a node mapping matrix multiplication
Node *n = arena_node_alloc(&meta_arena, 
                           (Tensor*[]){X, W},  // Inputs list
                           2,                  // Number of inputs
                           out,                // Output tensor
                           get_op_impl(MATMUL),// Operator kernel implementations
                           MATMUL,             // Operator enum type
                           (KernelParams){0, 0, 0.0f}); // Optional parameters
```

---

## 4. Scheduling & Executing Graphs

Once the graph nodes are linked, you compile and execute them using the graph `Scheduler`:

```c
// Initialize JIT lookup table cache with capacity for 16 graphs
JIT *jit_cache = jit_create(16);
Scheduler *scheduler = init_scheduler(jit_cache);

// Enable graph fingerprint caching
set_jit_mode(scheduler, true);

// 1. Zero out model gradient buffers
zero_grad_cuda(W);

// 2. Execute Forward Sweep
schedule(scheduler, n, FORWARD, &meta_arena);

// 3. Populate starting seed gradient on outputs
set_ones_grad(out);

// 4. Execute Backward Sweep
schedule(scheduler, n, BACKWARD, &meta_arena);

// 5. Update weights using a native parameter optimizer
SGD optimizer = arena_alloc_sgd(0.01f);
sgd_step_cuda(&optimizer, (Tensor*[]){W}, 1);

// 6. Release JIT compiler caches and structures
scheduler_release(scheduler);
```

---

## 5. Cleaning Up Arenas

To avoid memory leaks, reset or release memory pools periodically:

```c
// Clears transient segments (useful at the end of each training step)
arena_reset(&data_arena);

// Releases the entire memory space back to the operating system / driver
arena_release(&meta_arena);
arena_release(&data_arena);
```

---

## 6. Compiling & Linking Standalone Binaries

### Statically Linking Plast
You compile the core Plast library using the provided `Makefile`.

1. Build the library (creates `libplast.a`):
   ```bash
   make lib CUDA=1
   ```
   *(Omit `CUDA=1` to compile a CPU-only fallback target)*

2. Compile your C application and link the static binary, adding OpenMP flags and CUDA library locations:
   ```bash
   cc -O3 -I./include -o my_app main.c -L. -lplast -lm -lgomp -fopenmp -lcudart -L/usr/local/cuda/lib64
   ```

### Makefile Integration
For project builds, you can integrate compilation directly into a `Makefile`:

```makefile
CC   = cc
NVCC = nvcc
CFLAGS = -O3 -Wall -fopenmp -I./include -I/usr/local/cuda/include

my_app: main.c libplast.a
	$(CC) $(CFLAGS) -o $@ main.c -L. -lplast -lm -lgomp -lcudart -L/usr/local/cuda/lib64
```
