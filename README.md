# plast

A lightweight, high-performance deep learning engine built from scratch in C and CUDA with zero external runtime dependencies.

plast features a custom arena allocator, a dynamic autograd graph engine, a cached DAG scheduler, operator fusion patterns, and a clean PyTorch-compatible Python API.

---

## Philosophy

Deep learning engines should treat neural networks for what they are: **dataflow graphs of pure transformations**.

Most standard frameworks layer heavy, object-oriented abstractions (like leaky state registries or complex OOP hierarchies) over what is fundamentally a mathematical dataflow graph. While plast exposes a PyTorch-compatible interface for ease of use, its core is built around **functional dataflow pipelines**. You compose tensor transformations, and the scheduler handles the graph compilation, JIT optimization, and backend execution. 

This design keeps the codebase clean, ensures deterministic memory lifetimes, and moves optimization to where it belongs: the graph scheduler and the hardware kernels.

---

## Core Architectural Pillars

```
                     ┌───────────────────────────┐
                     │     Python Frontend       │  ← Sequential layers, trackers,
                     │  (plast/nn, plast/data)   │    and PyTorch-like interfaces
                     └─────────────┬─────────────┘
                                   │
                     ┌─────────────▼─────────────┐
                     │  Pybind11 Binding Layer   │  ← Exposes C-level Tensor, Arena,
                     │     (plast.plast_core)    │    and Scheduler structures
                     └─────────────┬─────────────┘
                                   │
                     ┌─────────────▼─────────────┐
                     │    DAG JIT Scheduler      │  ← Graph traversal, JIT cache via
                     │     (src/scheduler/)      │    fingerprints & operator fusion
                     └─────────────┬─────────────┘
                                   │
                     ┌─────────────▼─────────────┐
                     │      Arena Allocator      │  ← Memory pools separating weights
                     │       (src/core/)         │    (persistent) and activations (transient)
                     └─────────────┬─────────────┘
                                   │
                                   ├──────────────────────────────┐
                                   ▼                              ▼
                       ┌───────────────────────┐      ┌───────────────────────┐
                       │     CUDA Backend      │      │      CPU Backend      │
                       │   sm_80+ (Ampere+)    │      │  AVX Vectorized + OMP  │
                       └───────────────────────┘      └───────────────────────┘
```

1. **Dual Arena Allocator**: Tensors are allocated in pre-sized memory arenas. Memory is split into **persistent** spaces (for parameters/weights) and **transient** spaces (for activations and temporary gradients). This design eliminates memory fragmentation and avoids `malloc`/`free` calls during the training loop.
2. **DAG JIT Scheduler**: When execution passes (`FORWARD` or `BACKWARD`) are called, the scheduler traverses the active computation graph. It generates a topological plan, caches it using a fast structural fingerprint, and matches subgraphs against operator fusion patterns (e.g., merging `Matmul` -> `Add` -> `ReLU` into a single operation).
3. **Hardware Dispatch**: Dispatches optimized C code to either CPU Vectorized (AVX + OpenMP) or GPU CUDA (tiled shared-memory architectures targeting sm_80 and above) backend kernels.

---

## Current Project Phase & Feature Matrix

| Subsystem | Feature | Status | Details |
|---|---|---|---|
| **Core** | Autograd | **Done** | Dynamic DAG building with automatic topological sort and correct gradient accumulation. |
| | Strided Tensors | **Done** | N-dimensional tensor layout supporting views, transpositions, and reshaping without memory copies. |
| | Arena Allocator | **Done** | Separation of persistent (weights) and transient (graph) data pools. |
| **Backends** | CPU Kernels | **Done** | AVX-accelerated element-wise, reduction, and convolutional operations with OpenMP parallelism. |
| | CUDA Kernels | **Done** | Tiled matrix multiplication, warp-level reductions, and fast `im2col` convolutions. |
| | Operator Fusion | **Done** | In-kernel CUDA fusions for standard sequences: `Matmul + ReLU`, `Matmul + Bias + ReLU`, and `Conv2d + ReLU`. |
| **Scheduler** | Graph JIT | **Done** | Structure fingerprint caching to skip DAG scheduling analysis on repeated execution runs. |
| **High-level API**| Classic PyTorch Layering| **Done** | Familiar interfaces: `nn.Module`, `nn.Sequential`, layers, losses (`MSELoss`, `CrossEntropyLoss`), and optimizers (`Adam`, `AdamW`, `SGD`). |
| | Data Loader | **Done** | Native C-backed `TensorDataset` and `DataLoader` supporting batched loading, shuffling, and automatic device staging. |
| | Pipeline DSL | **Roadmap** | Fluent functional composition API (`pipe(...)`) with lazy execution paths. |
| | Experiment Tracking | **Done** | Out-of-the-box run logging, YAML-serialized parameters, performance charts, and checkpoint savings. |

---

## Python Quickstart

### 1. Installation

Build the static engine and install the Python bindings in editable mode:

```bash
# Clone the repository
git clone https://github.com/Vixel2006/plast.git
cd plast

# Build and install the bindings in your environment
pip install -e . --no-build-isolation
# Or using uv:
uv pip install -e . --no-build-isolation
```

### 2. XOR Model Example

A complete training step using the high-level Python API, memory management, and JIT caching:

```python
import plast as p
import numpy as np

# 1. Initialize memory pools (default size: 64MB metadata, 512MB data)
p.init_arenas(device=p.Device.CPU)

# 2. Define a standard Sequential model
model = p.nn.Sequential(
    p.nn.Linear(2, 8),
    p.nn.ReLU(),
    p.nn.Linear(8, 1)
)

loss_fn = p.nn.MSELoss()
optimizer = p.optim.Adam(model.parameters(), lr=0.01)

# 3. Prepare dataset arrays
x_data = p.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = p.tensor([[0], [1], [1], [0]])

# 4. Use @p.jit to cache graph traversal and optimization planning
@p.jit
def train_step(x, y):
    optimizer.zero_grad()
    
    # Forward pass
    preds = model(x)
    loss = loss_fn(preds, y)
    
    # Backward pass & Optimizer updates
    loss.backward()
    optimizer.step()
    return loss

# 5. Training loop
for epoch in range(1001):
    loss = train_step(x_data, y_data)
    
    # Reset intermediate activations/gradients in the transient arena
    p.reset_transient_arenas()
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")
```

> [!TIP]
> **Why call `p.reset_transient_arenas()`?**  
> Intermediate activation and gradient tensors are allocated within a fast transient memory arena. Resetting this arena at the end of each iteration clears these temporary allocations, maintaining a constant memory footprint and preventing overhead during training.

---

## Native C API

Plast is written as a pure, lightweight C library. The C API lets you initialize execution contexts, build graphs, and train models at the bare-metal level:

```c
#include "core/arena.h"
#include "core/tensor.h"
#include "core/node.h"
#include "scheduler/scheduler.h"

int main() {
    // 1. Create arenas (Metadata is CPU-only, Data is CUDA/GPU-bound)
    Arena meta_arena = arena_create(Mib(10), CPU);
    Arena data_arena = arena_create(Mib(100), CUDA);

    // 2. Allocate strided tensors directly inside the memory pool
    Tensor *X   = init(&meta_arena, &data_arena, CUDA, FLOAT32, (u64[]){4, 2}, 2, false, NULL);
    Tensor *W   = init(&meta_arena, &data_arena, CUDA, FLOAT32, (u64[]){2, 8}, 2, true, rand_init);
    Tensor *out = init(&meta_arena, &data_arena, CUDA, FLOAT32, (u64[]){4, 8}, 2, true, NULL);

    // 3. Attach operations as nodes in the graph
    Node *matmul_node = arena_node_alloc(&meta_arena, (Tensor*[]){X, W}, 2, out, 
                                         get_op_impl(MATMUL), MATMUL, (KernelParams){0, 0, 0.0f});

    // 4. Set up the execution scheduler and its JIT plan cache
    JIT *jit_cache = jit_create(16);
    Scheduler *scheduler = init_scheduler(jit_cache);

    // 5. Run forward and backward sweeps
    schedule(scheduler, matmul_node, FORWARD, &meta_arena);
    set_ones_grad(out);
    schedule(scheduler, matmul_node, BACKWARD, &meta_arena);

    // 6. Free execution structures and reset memory pools
    scheduler_release(scheduler);
    arena_release(&meta_arena);
    arena_release(&data_arena);
    return 0;
}
```

See [main.c](./main.c) for a complete native C training example on XOR data.

---

## Repository Structure

```
plast/
├── include/                 # Public C headers
│   ├── core/                # Tensor layouts, autograd nodes, memory pools
│   ├── kernels/             # Backend operations declarations
│   ├── optimizers/          # Updates & parameter stepping parameters
│   └── scheduler/           # JIT graph caches and pattern matching fusions
├── src/                     # C/CUDA sources
│   ├── core/                # Autograd backward sweeps and arena engines
│   ├── kernels/cpu/         # CPU-optimized operations (OpenMP, AVX)
│   ├── kernels/cuda/        # Fused and tiled CUDA implementations (sm_80+)
│   ├── optimizers/          # SGD, Adam, AdamW logic
│   └── python/              # Pybind11 binding interfaces
├── plast/                   # High-level Python package
│   ├── data/                # Dataset & DataLoader modules
│   ├── experiment/          # Configuration trackers, checkpointing, and metric savers
│   ├── nn/                  # PyTorch-compatible modules, Sequential models, and layers
│   └── optim/               # High-level parameter stepping & schedulers
└── tests/                   # Extensive PyTest suite verifying ops, grads, and e2e paths
```

---

## Roadmap

- [x] **CUDA Kernel Fusions** — Fused CUDA implementations for standard blocks (Matmul+ReLU, Conv+ReLU).
- [x] **JIT Graph Fingerprinting** — Schema verification caching to avoid re-scheduling identical structures.
- [ ] **Functional Pipeline DSL** — High-level, lazy-evaluating functional piping syntax (`pipe()`) for Python.
- [ ] **Mixed Precision** — Native FP16/BF16 tensor cores support for modern GPU architectures.
- [ ] **Distributed Training** — Multi-GPU execution via NCCL and MPI.

---

## Contributing

We welcome performance-driven contributions! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for code formatting guidelines (clang-format/ruff), testing standards, and compiler checks.
