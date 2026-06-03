# plast

a high-performance deep learning engine from scratch in C and CUDA.

zero dependencies. full autograd. tiled Ampere-optimized CUDA kernels. arena-allocated memory.

neural networks are data transformation pipelines. plast treats them that way.

## philosophy

most frameworks model a network as a tree of objects. `nn.Module` subclasses, `forward()` methods, parameter registries — a leaky OOP abstraction over what is fundamentally a dataflow graph.

plast has that too (see [classic api](#classic-api)). but the real path is the **functional pipeline**:

```
input → normalize → relu → matmul(W1) → relu → matmul(W2) → softmax → loss
```

each stage is a pure function over tensors. compose them lazily. the scheduler sees the whole graph and JIT-compiles fused kernels for it. no python overhead between layers. no kernel launch overhead between fusable ops.

zero-cost abstractions are a compiler problem, not an API problem.

## quickstart

```sh
git clone https://github.com/Vixel2006/plast.git
cd plast
pip install -e . --no-build-isolation
```

```python
import plast as p
from plast.experiment import ExperimentConfig, ExperimentTracker

# functional pipeline — the pro path
def model(x, W1, b1, W2, b2):
    return x @ W1 + b1 |> p.nn.functional.relu |> p.nn.functional.linear(W2, b2)

# or the classic api — for noobs
net = p.nn.Sequential(
    p.nn.Linear(784, 256),
    p.nn.ReLU(),
    p.nn.Linear(256, 10),
)
```

## architecture

plast uses a clean layered design. the scheduler orchestrates a DAG of tensor ops and dispatches to hardware-specific backends. a JIT compiler fuses adjacent ops into single CUDA kernels where possible.

```
                        [ Python API ]
                              │
                              ▼
                    ┌─────────────────────┐
                    │   pipeline composer │  ← functional pipelines, lazy graphs
                    │   (plast/pipeline/) │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │   scheduler + JIT   │  ← graph fusion, kernel compilation
                    │   (src/scheduler/)  │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │   tensor / memory   │  ← arena allocator, stride-aware views
                    │   (src/tensor/)     │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │   hardware backends │  ← abstract op dispatch
                    │   (src/backend/)    │
                    └──────────┬──────────┘
                               │
                   ┌───────────┴───────────┐
                   ▼                       ▼
            ┌──────────────┐      ┌──────────────┐
            │cuda kernels  │      │  cpu / simd  │
            │  sm_80+      │      │  avx+omp     │
            └──────────────┘      └──────────────┘
```

## features

| module | what it does |
|--------|-------------|
| **autograd** | dynamic DAG with topological sort, correct gradient accumulation for reused nodes |
| **tensor** | n-dimensional strided tensors, broadcasting, views without data copy |
| **cuda kernels** | sm_80-optimized tiled matmul (32×32), fused element-wise, warp-level reductions, im2col conv2d |
| **jit compiler** | (in development) runtime CUDA kernel fusion via NVRTC — fuses adjacent ops into single launch |
| **scheduler** | DAG-graph traversal with integrated JIT — whole-pipeline fusion planning |
| **functional pipeline** | lazy data-transformation composition `a |> f |> g`, zero overhead between stages |
| **classic api** | `nn.Module`, `nn.Sequential`, optimizers, dataloaders — pytorch-compatible ergonomics |
| **arena allocator** | pre-allocates full graph memory. zero malloc/free during training. deterministic lifetime |
| **experiment tracking** | versioned runs, YAML configs/metrics, automatic checkpointing, best-run aggregation |

## experiment tracking

experiments are first-class citizens. each run is auto-versioned under `experiments/{name}/run_{NNN}/`:

```python
config = ExperimentConfig(
    name="mlp_mnist",
    model={"hidden": 256, "dropout": 0.2},
    training={"lr": 1e-3, "epochs": 50},
    device="cuda",
)
tracker = ExperimentTracker(config)

for epoch in range(50):
    loss = train_epoch(model, data)
    tracker.log_epoch(epoch, {"train_loss": loss})

tracker.finish()
```

produces:

```
experiments/mlp_mnist/
├── summary.yaml            ← best run across all runs
├── run_001/
│   ├── config.yaml         ← frozen config
│   ├── metrics.yaml        ← per-epoch metrics
│   └── checkpoints/
│       └── best_model.npz  ← best weights
├── run_002/
│   └── ...
```

## the pipeline approach (vs oop modules)

the classic `nn.Module` API exists and works. it's fine for quick experiments and for people who think in layers.

the functional pipeline is different. you define your forward pass as a composition of pure tensor functions:

```python
from plast.pipeline import pipe

forward = pipe(
    lambda x: p.nn.functional.linear(x, W1, b1),
    p.nn.functional.relu,
    lambda x: p.nn.functional.linear(x, W2, b2),
    p.nn.functional.softmax,
)
```

the `pipe` object is lazy. nothing executes until you call it. the scheduler intercepts the composition, builds a DAG, and hands it to the JIT compiler. fusable ops are merged into a single CUDA kernel. non-contiguous data is packed once. the arena is pre-sized for the exact pipeline shape.

what you get:
- **fused kernels** — relu + matmul becomes one launch instead of two
- **lazy allocation** — arena sized from the static graph, not on every `forward()`
- **zero python overhead** — the entire pipeline runs through compiled code once the graph is built

## cuda kernels

all kernels target **sm_80** (Ampere) and above. currently shipping:

| kernel | approach | status |
|--------|----------|--------|
| matmul | tiled 32×32, shared memory, NT/TN variants, batch support | done |
| conv2d | im2col + matmul, col2im backward | done |
| element-wise | contiguous fast path + broadcast fallback, fused variants planned | done |
| reductions | warp-level tree reduction, shared-memory dim reduction | done |
| optimizers | SGD, Adam, AdamW fused kernels | done |

## c api

plast is a C library first. the python bindings are a thin pybind11 layer. if you want to embed the engine in a C or C++ project:

```c
#include "arena.h"
#include "tensor.h"
#include "graph.h"

Arena arena = arena_create(1024 * 1024 * 1024, DEVICE_CUDA);
Tensor* x = tensor_create(&arena, shape, 2, FLOAT);
Tensor* y = tensor_create(&arena, shape_out, 2, FLOAT);
Node*  n = matmul_node(&arena, x, y, NULL);
forward(n);
backward(n);
arena_release(&arena);
```

## roadmap

- [ ] **Optimizing CUDA Kernels** — Optimizing our slow cuda kernels
- [ ] **JIT compiler** — NVRTC-based kernel fusion in the scheduler
- [ ] **fused ops** — matmul+relu, conv+bn+relu as single kernels
- [ ] **functional pipeline builder** — `pipe()` API with lazy DAG construction
- [ ] **mixed precision** — FP16/BF16 on tensor cores
- [ ] **distributed** — MPI + NCCL for multi-GPU training
- [ ] **python bindings for pipeline** — bring the pro path to python

## contributing

prs welcome. keep it zero-dependency. keep it fast.
