# Plast Technical Documentation

Welcome to the Plast technical documentation. This sitemap indexes the architecture, design patterns, and programming interfaces of the Plast deep learning engine.

---

## 1. Documentation Index

* 🚀 **[Getting Started](./getting_started.md)**: Environment setup, compiling Python bindings, and training a simple XOR MLP model.
* 🏗️ **[Engine Architecture](./architecture.md)**: Deep dive into the custom memory Arenas (persistent vs. transient), strided tensor layouts, DAG autograd traversal, and the graph scheduler.
* 🐍 **[Python API Reference](./python_api.md)**: Detailed classes, factory functions, mathematical operators, modules, optimizers, and schedulers in the python frontend.
* 🔌 **[C API Guide](./c_api.md)**: Working with memory pools, creating tensors, creating nodes, running schedule passes, and compiling/linking standalone binaries in C.
* ⚡ **[Adding Custom Kernels & Fusions](./custom_kernels.md)**: Developer guide for registering custom CPU (AVX) and CUDA (sm_80+) kernels, and writing manual operator fusions in the scheduler.

---

## 2. Core Concepts Reference

If you are new to Plast, these are the key architectural decisions that distinguish Plast from standard frameworks like PyTorch or JAX:

### The Arena Allocation Pattern
To avoid system allocation (`malloc`/`free`) overhead during training loops, Plast pre-allocates contiguous memory pools.
```python
import plast as p

# 1. Initialize persistent and transient pools
p.init_arenas(device=p.Device.CUDA)

# ... training step ...

# 2. Reset intermediate activations & gradients
p.reset_transient_arenas()
```
* **Persistent Arena**: Retains model weights, biases, and running parameters.
* **Transient Arena**: Retains intermediate outputs and backward gradients. This arena is cleared at the end of each training loop iteration using `reset_transient_arenas()`.

### Caching JIT Graph Scheduler
The execution paths are compiled and cached on their first sweep. Tensors are realized dynamically via:
```python
# Explicit execution
p.forward(tensor)

# Implicit execution (triggered automatically)
values = tensor.numpy()
```
Decorating step functions with `@p.jit` enables structural hashing, caching the topological ordering and scheduling fusions for subsequent passes.
