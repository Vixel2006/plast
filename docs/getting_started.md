# Getting Started with Plast

Welcome to Plast! This guide helps you install the library, understand its custom arena allocation model, and train your first neural network model in Python.

---

## 1. What is Plast?

Plast is a high-performance deep learning engine built from scratch in C and CUDA, designed to be lightweight, efficient, and completely free of heavy framework overhead.

### Why choose Plast?
* **Zero Runtime Allocation Overhead**: Avoids system memory allocators during the training loop via custom pre-allocated memory pools.
* **Topological Graph Scheduling**: Builds a dynamic computation DAG and optimizes it via pattern-matching operator fusion and JIT traversal caching.
* **Familiar PyTorch-compatible API**: Exposes familiar objects (`nn.Module`, `data.DataLoader`, `optim.Adam`, and autograd `.backward()`) for rapid onboarding.

---

## 2. Installation

Plast compiles its C++ library core and links it with Python bindings via [pybind11](https://github.com/pybind/pybind11).

### Prerequisites
* **C++ Compiler**: `gcc` or `clang` with OpenMP support.
* **CUDA Toolkit**: Optional, required for GPU acceleration (requires `nvcc` and an NVIDIA Ampere or newer GPU with `sm_80+` support).
* **Python**: Version 3.8 or newer.
* **uv** or **pip** package managers.

### Build and Install

1. Clone the repository:
   ```bash
   git clone https://github.com/Vixel2006/plast.git
   cd plast
   ```

2. Compile and install in editable mode:
   ```bash
   pip install -e . --no-build-isolation
   ```
   *If you are using the modern, faster `uv` package manager:*
   ```bash
   uv pip install -e . --no-build-isolation
   ```

3. Run the unit test suite to verify compile flags and target hardware configurations:
   ```bash
   pytest tests/ -v
   ```

---

## 3. XOR Classifier Tutorial

Training a model in Plast uses a familiar training flow. Because Plast is built around an **arena allocator** and a **JIT scheduler**, we specify memory pools at initialization and reset activation buffers at the end of each training loop iteration.

Here is a complete, working script that trains a Multi-Layer Perceptron (MLP) on the classic XOR problem:

```python
import plast as p
import numpy as np

def train_xor():
    # 1. Initialize Arenas
    # Pre-allocates memory arenas on the target device.
    p.init_arenas(device=p.Device.CPU)

    # 2. Define the Model
    # A standard Sequential MLP container
    model = p.nn.Sequential(
        p.nn.Linear(2, 8),
        p.nn.ReLU(),
        p.nn.Linear(8, 1)
    )

    loss_fn = p.nn.MSELoss()
    optimizer = p.optim.Adam(model.parameters(), lr=0.01)

    # 3. Prepare Tensors
    # Tensors are created directly inside the memory arenas.
    x_data = p.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], device=p.Device.CPU)
    y_data = p.tensor([[0], [1], [1], [0]], device=p.Device.CPU)

    # 4. Define the training step decorated with @p.jit
    # The JIT compiler structural-hashes the graph on the first run and caches 
    # the execution schedule to skip planning overhead in subsequent epochs.
    @p.jit
    def train_step(x, y):
        optimizer.zero_grad()
        
        # Forward pass (creates nodes in the DAG)
        preds = model(x)
        loss = loss_fn(preds, y)
        
        # Backward pass & Optimizer updates
        loss.backward()
        optimizer.step()
        return loss

    # 5. Training loop
    print("Training MLP XOR model...")
    for epoch in range(1001):
        loss = train_step(x_data, y_data)
        
        # CRITICAL: Reset transient memory pools to free activation/gradient buffers
        p.reset_transient_arenas()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

    # 6. Evaluate Model Predictions
    test_preds = model(x_data)
    
    # Retrieve final values as a NumPy array (automatically executes the forward path)
    predictions = test_preds.numpy()
    print("\nFinal Model Predictions:")
    print(predictions)

if __name__ == "__main__":
    train_xor()
```

---

## 4. Key Differences vs. PyTorch

While Plast's syntax is heavily inspired by PyTorch, there are structural differences under the hood:

### 1. Lazy Computation Graphs
In PyTorch, operations execute eagerly. In Plast, calling mathematical operations on a tensor adds a node to a Directed Acyclic Graph (DAG). Computation is only realized when:
* `.forward()` (or `p.forward(tensor)`) is called on the tensor.
* `.numpy()` is called to retrieve the numpy array view (shares CPU memory where possible).
* `.item()` is called to retrieve a python scalar.
* `.backward()` is called (which automatically triggers `.forward()` first to evaluate the values).

### 2. The Dual-Arena Memory Model
PyTorch uses dynamic GPU/CPU allocators (like `cudaMalloc` caching). Plast avoids allocation overhead entirely by splitting memory into two fixed-size pools:
* **Persistent Arena**: Retains model parameters, weights, biases, and running stats that must persist throughout training.
* **Transient Arena**: Stores intermediate outputs, activations, and backpropagation gradients. Because these are temporary, you must call `p.reset_transient_arenas()` at the end of each training loop iteration to clear this arena.

> [!TIP]
> **Why clean the transient arena?**  
> If you omit `p.reset_transient_arenas()`, new activations and gradients will append to the pool until the transient arena runs out of space. Resetting is a zero-overhead pointer rollback that keeps memory usage constant.
