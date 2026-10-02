# Engine Architecture

This document provides a deep dive into the systems-level design of Plast. Plast is structured to minimize memory overhead, eliminate CPU scheduling bottlenecks, and maximize execution throughput.

---

## 1. Arena Memory Management

Modern deep learning frameworks spend significant runtime overhead in GPU driver allocations (`cudaMalloc` / `cudaFree`) and garbage collection. Plast bypasses this entirely using a **Dual Arena Memory Model**.

```
                   ┌─────────────────────────────────────────┐
                   │           Model Memory Pool             │
                   └────────────────────┬────────────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
    ┌───────────────────────────┐                 ┌───────────────────────────┐
    │     Persistent Arena      │                 │      Transient Arena      │
    │   (Weights & Param State) │                 │   (Activations & Grads)   │
    ├───────────────────────────┤                 ├───────────────────────────┤
    │  - Lifespan: Session      │                 │  - Lifespan: Epoch/Step   │
    │  - No automatic resets    │                 │  - Periodically cleared   │
    └───────────────────────────┘                 └───────────────────────────┘
```

When Plast initializes (`init_arenas`), it creates two independent memory regions:
1. **Persistent Arena**: Houses parameters, weight matrices, optimizer variables (such as momentum and adaptive moment buffers), and constants. Tensors allocated here are never automatically freed during execution loops.
2. **Transient Arena**: Houses intermediate activations, dynamic graph nodes, and backpropagation gradients. Because these values are only needed for a single training step, this entire arena is cleared when calling `reset_transient_arenas()`, resetting the allocation pointer to zero in constant time ($O(1)$) and avoiding heap churn.

### Device Allocation
Each arena is configured to run on either the `CPU` (using standard heap segments) or `CUDA` (using GPU device pointers). Plast's core scheduler staging automatically copies metadata arrays to CPU memory while targeting computation buffers to the GPU.

---

## 2. Stride-Aware N-Dimensional Tensors

Plast represents tensors using the `Tensor` struct. Each tensor stores a pointer to its data, a rank (`ndim`), shapes, and strides:

```c
typedef struct Tensor {
  struct Tensor *grad;    // Gradient tensor of matching shape
  struct Node *creator;   // Op node that produced this tensor
  void *data;             // Pointer inside the CPU or CUDA Arena
  u64 shape[MAX_NDIM];    // Dimension lengths
  u64 strides[MAX_NDIM];  // Memory offsets per dimension
  u64 ndim;               // Number of dimensions
  DTYPE dtype;            // Data type (e.g. FLOAT32)
  DEVICE device;          // Execution device (CPU or CUDA)
  bool requires_grad;     // Gradient tracking flag
} Tensor;
```

### Strided Memory Views
By using strides, operations like transposing, reshaping, slicing, or squeezing can be performed in $O(1)$ time without copying the underlying data:
* **Reshape/View**: Computes new strides for contiguous blocks.
* **Transpose**: Swaps the shapes and strides of the target axes.
* **Non-Contiguous Operations**: When passing data to CUDA kernels that require contiguous layout (like matrix multiplication), the scheduler staging copies and packs the memory into contiguous temporary buffers before execution.

---

## 3. The DAG Autograd Engine

The computation graph is modeled as a Directed Acyclic Graph (DAG). 
* **Leaf Tensors**: User inputs or model parameters that have no `creator`.
* **Intermediate Tensors**: Created by operators, containing a pointer to the `Node` that generated them.
* **Nodes**: Retain pointers to input tensors, the output tensor, the specific operation logic (`Op`), and any hyperparameters (`KernelParams`).

```
    [Tensor: X] ──┐
                  ├─► [Node: Matmul] ─► [Tensor: Outputs]
    [Tensor: W] ──┘
```

### Forward Staging & Backward Propagation
When you trigger an execution pass, Plast uses the scheduler to traverse the graph:
1. **Forward Pass**: Recursively visits dependencies to compute topological order. It then executes each node's forward kernel, allocating intermediate output buffers inside the active transient arena.
2. **Backward Pass**: Traverses the graph in reverse topological order. It populates gradients using backward kernels, carrying accumulation chains safely across shared node paths.

---

## 4. JIT Scheduler & Manual Operator Fusion

 Plast uses a JIT graph scheduler (`Scheduler`) to analyze the DAG and plan optimizations before sending operations to hardware kernels.

### Graph Fingerprinting
Building execution plans and sorting graphs topologically on every forward/backward sweep introduces CPU overhead. Plast eliminates this using structural fingerprinting:
1. When JIT mode is enabled (`set_jit_mode(true)`), calling `schedule` computes a recursive structural hash of the tensor DAG based on shape dimensions, data types, and node operations.
2. The scheduler looks up this fingerprint in a high-performance hash table cache (`JIT` cache).
3. If matched, it reuses the pre-compiled topological order and execution path, bypassing traversal logic entirely.

### Manual Operator Fusion
To minimize kernel launch overhead and GPU memory bus bottleneck, the scheduler scans subgraphs for standard sequence patterns and substitutes them with optimized **fused kernels**:

```
      Matmul Node ──► Add Node ──► Leaky ReLU Node
                          │
                          ▼
            [Fused Matmul-Bias-ReLU Node]
```

#### Supported Fusion Patterns
* **Matmul + LeakyReLU** (`FUSION_MATMUL_RELU`): Replaced by a single launch of `matmul_relu.cu`.
* **Matmul + Add (Bias) + LeakyReLU** (`FUSION_MATMUL_BIAS_RELU`): Replaced by a single launch of `matmul_bias_relu.cu`.
* **Conv2d + LeakyReLU** (`FUSION_CONV_RELU`): Replaced by a single launch of `conv_relu.cu`.

These replacements are performed dynamically during the schedule planning phase. By merging element-wise operations directly into matrix and convolution updates, Plast saves expensive intermediate read/write cycles to global GPU memory.
