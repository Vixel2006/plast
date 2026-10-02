# Adding Custom Kernels & Fusions

Plast relies on hand-optimized, manual CPU and CUDA kernels for execution. This developer guide walks you through writing a new operation, registering its hardware kernels, exporting it via Python bindings, and integrating it into the scheduler's manual operator fusion pass.

---

## 1. Writing the Hardware Kernels

Every operation must implement a forward and backward kernel signature for both **CPU** (using C with AVX/OpenMP) and **CUDA** (using `.cu` files).

### Operation Kernel Signature
Operation kernels match the standard signatures declared in `include/core/op.h`:

```c
typedef void (*ForwardKernel)(const Tensor *inputs[], Tensor *output, KernelParams params);
typedef void (*BackwardKernel)(const Tensor *inputs[], const Tensor *output, Tensor *grads[], KernelParams params);
```

### Example: Custom Exponential operation (`exp`)

#### CPU Kernel (`src/kernels/cpu/unary_ops.c`)
On the CPU, loop over elements. If memory layout is contiguous, use a fast single-instruction, multiple-data (SIMD) sweep:

```c
#include <math.h>

void cpu_exp_forward(const Tensor *inputs[], Tensor *output, KernelParams params) {
  const Tensor *x = inputs[0];
  float *x_data = (float *)x->data;
  float *y_data = (float *)output->data;
  u64 n = numel(x);

  #pragma omp parallel for
  for (u64 i = 0; i < n; ++i) {
    y_data[i] = expf(x_data[i]);
  }
}
```

#### CUDA Kernel (`src/kernels/cuda/unary_ops.cu`)
On the GPU, map elements to CUDA threads:

```cuda
__global__ void exp_forward_cuda_kernel(const float *x, float *y, u64 n) {
  u64 idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = expf(x[idx]);
  }
}

extern "C" void cuda_exp_forward(const Tensor *inputs[], Tensor *output, KernelParams params) {
  const Tensor *x = inputs[0];
  u64 n = numel(x);
  u64 threads = 256;
  u64 blocks = (n + threads - 1) / threads;

  exp_forward_cuda_kernel<<<blocks, threads>>>((float *)x->data, (float *)output->data, n);
  cudaDeviceSynchronize();
}
```

---

## 2. Registering the Operation in the Core

### Step A: Update the `OP_TYPE` Enum
Add the new operation to the `OP_TYPE` enum inside `include/core/op.h`:

```c
typedef enum OP_TYPE {
  ADD,
  SUB,
  // ...
  EXP,   // Our new operation
  // ...
} OP_TYPE;
```

### Step B: Hook into `get_op_impl`
Register the C/CUDA functions inside the op registry in `src/core/op.c`:

```c
#include "kernels/cpu/unary_ops.h"
#ifdef CUDA_AVAILABLE
#include "kernels/cuda/unary_ops.h"
#endif

Op get_op_impl(OP_TYPE type) {
  switch(type) {
    case EXP:
      return (Op){
        .cpu_forward = cpu_exp_forward,
        .cpu_backward = cpu_exp_backward,
#ifdef CUDA_AVAILABLE
        .cuda_forward = cuda_exp_forward,
        .cuda_backward = cuda_exp_backward,
#endif
      };
    // ...
  }
}
```

---

## 3. Exporting to Python Bindings

To make your operation accessible in Python:

### Step A: Register the OpType Enum Value
Add the new enum value inside [src/python/bindings.cpp](file:///home/vixel/code/plast/src/python/bindings.cpp):

```cpp
py::enum_<OP_TYPE>(m, "OpType")
    .value("ADD", ADD)
    // ...
    .value("EXP", EXP) // Export to python bindings
    .export_values();
```

### Step B: Add the Method in Python
Hook the operation method onto the `Tensor` class inside [plast/tensor.py](file:///home/vixel/code/plast/plast/tensor.py):

```python
class Tensor:
    # ...
    def exp(self) -> "Tensor":
        """Compute the element-wise exponential of the tensor."""
        # _run_op automatically allocates memory and adds a node to the active DAG
        return _run_op([self], OpType.EXP, list(self.shape))
```

---

## 4. Writing Operator Fusion Logic

Plast uses manual fused kernels (e.g. `matmul_relu.cu`) to merge consecutive operations into a single kernel execution.

If you are adding a new operation that can be fused with adjacent operations, implement the pattern matching and node replacement inside [src/scheduler/fusion.c](file:///home/vixel/code/plast/src/scheduler/fusion.c).

### Pattern Matching Walkthrough

1. **Detect Pattern**: Scan the computational graph (in topological order) to locate the sequence. For example, matching `EXP` -> `ADD` (bias):
   ```c
   static u32 try_match_ending_at(DAG *dag, u32 node_idx, FusionMatch *match) {
     Node *n = dag->nodes[node_idx];
     
     if (n->op_type == ADD) {
       Node *producer = n->inputs[0]->creator;
       if (producer && producer->op_type == EXP) {
         match->type = FUSION_EXP_ADD;
         match->nodes = malloc(2 * sizeof(Node *));
         match->nodes[0] = producer; // EXP
         match->nodes[1] = n;        // ADD
         match->num_nodes = 2;
         match->valid = true;
         return 1;
       }
     }
     return 0;
   }
   ```

2. **Define Fused Op Type**: Define a new combined op enum type (e.g., `EXP_ADD` inside `OP_TYPE`), and assign its forward/backward hooks to a hand-written fused C/CUDA kernel (`exp_add.cu`).

3. **Perform DAG Replacement**:
   Update `fusion_apply` to replace the matched nodes with a single fused node containing the combined inputs and outputs:
   ```c
   case FUSION_EXP_ADD: {
     Node *exp_node = match->nodes[0];
     Node *add_node = match->nodes[1];
     
     fused->op_type = EXP_ADD;
     fused->op = get_op_impl(EXP_ADD);
     fused->inputs = exp_node->inputs;     // Take original EXP inputs
     fused->num_inputs = exp_node->num_inputs;
     fused->output = add_node->output;     // Route directly to output
     break;
   }
   ```
   The fusion manager will automatically collapse the matching indices, update tensor pointers, and schedule the fused kernel in a single launch.
