# Python API Reference

This document provides a comprehensive API reference for the high-level Python package `plast`. 

Plast's Python API is designed to mirror PyTorch for familiarity, but runs on a custom high-performance C/CUDA core with deterministic memory arenas and a topological JIT compilation scheduler.

---

## 1. Memory & Arena Management

To eliminate runtime memory fragmentation and allocation overhead during training loops, Plast uses custom memory arenas.

### `plast.init_arenas`
```python
plast.init_arenas(meta_size_mb=64, data_size_mb=512, device=Device.CPU)
```
Initializes the global persistent and transient memory pools. Must be called once before constructing any tensors or model components.
* **`meta_size_mb`** (int): Allocation size for tensor descriptors and autograd graph nodes (CPU only).
* **`data_size_mb`** (int): Allocation size for the main float32 data buffers.
* **`device`** (plast.Device): Allocation backend target (`plast.Device.CPU` or `plast.Device.CUDA`).

### `plast.arena_scope`
```python
with plast.arena_scope(meta_size_mb=10, data_size_mb=100, device=Device.CPU):
    # scope execution block
```
Context manager that initializes and runs a block of code inside a temporary memory arena. When the scope is exited, all metadata and data allocations are immediately released and the arenas are destroyed.

### `plast.reset_transient_arenas`
```python
plast.reset_transient_arenas()
```
Manually resets the transient arena memory pool (clearing temporary activations, forward DAG outputs, and backward gradients). 
> [!IMPORTANT]
> Call this at the end of each training batch iteration to maintain a constant memory footprint and prevent memory exhaustions.

---

## 2. Tensor Creation & Factory Functions

Tensors in Plast are multi-dimensional arrays stored in the current memory arenas.

### Primary Factories
* **`plast.tensor(data, device=Device.CPU, dtype=DType.Float32, requires_grad=False, persistent=False)`**: Creates a tensor from Python lists, nested tuples, or NumPy arrays. If `persistent=True`, the tensor is allocated in the persistent arena (e.g. for parameters) instead of the transient arena.
* **`plast.Parameter(data, *, device=Device.CPU, dtype=DType.Float32, requires_grad=True)`**: Creates a trainable parameter. Shortcut for `plast.tensor(..., requires_grad=True, persistent=True)`.

### Uniform Initializers
* **`plast.zeros(shape, *, device=Device.CPU, requires_grad=False)`**: Returns a tensor of shape filled with `0.0`.
* **`plast.ones(shape, *, device=Device.CPU, requires_grad=False)`**: Returns a tensor of shape filled with `1.0`.
* **`plast.zeros_like(t, *, requires_grad=False)`**: Returns a tensor of zeros matching the shape and device of tensor `t`.
* **`plast.ones_like(t, *, requires_grad=False)`**: Returns a tensor of ones matching the shape and device of tensor `t`.
* **`plast.full(shape, fill_value, *, device=Device.CPU, requires_grad=False)`**: Returns a tensor filled with `fill_value`.
* **`plast.full_like(t, fill_value, *, requires_grad=False)`**: Returns a tensor filled with `fill_value` matching the shape and device of `t`.

### Random Initializers
* **`plast.randn(shape, *, device=Device.CPU, requires_grad=False)`**: Returns a tensor filled with random values sampled from a standard normal distribution $\mathcal{N}(0, 1)$.
* **`plast.rand(shape, *, device=Device.CPU, requires_grad=False)`**: Returns a tensor filled with uniform random values in the interval $[0, 1)$.
* **`plast.randint(low, high, shape, *, device=Device.CPU)`**: Returns a float32 tensor containing integer samples uniformly selected from $[low, high)$.
* **`plast.eye(n, *, device=Device.CPU, requires_grad=False)`**: Returns a 2-D identity matrix of shape `[n, n]`.

### Sequences
* **`plast.arange(start, stop=None, step=1, *, device=Device.CPU)`**: Returns a 1-D tensor with evenly spaced values within the interval $[start, stop)$ with spacing `step`.
* **`plast.linspace(start, end, steps, *, device=Device.CPU)`**: Returns a 1-D tensor with `steps` linearly spaced values between `start` and `end` inclusive.

---

## 3. High-Level Tensor Utilities

* **`plast.cat(tensors, dim=0)`**: Concatenates a sequence of tensors along an existing dimension `dim`.
* **`plast.stack(tensors, dim=0)`**: Stacks a sequence of tensors along a **new** dimension `dim`.
* **`plast.where(condition, x, y)`**: Selects elements from `x` or `y` based on `condition` element-wise.
* **`plast.clip(t, min_val=None, max_val=None)`** (or **`clamp`**): Clamps all elements of tensor `t` to the range $[min\_val, max\_val]$ in-place style.
* **`plast.save(model_or_state, path)`**: Saves a model's state dict or any key-value dictionary of numpy arrays to a `.npz` file.
* **`plast.load(model, path, strict=True)`**: Loads weights from a `.npz` file into the model in-place.
* **`plast.clip_grad_norm_(parameters, max_norm, norm_type=2.0)`**: Scales the gradients of an iterable of parameters in-place so that their combined $p$-norm does not exceed `max_norm`. Returns the total norm.
* **`plast.manual_seed(seed)`**: Sets the random seed for NumPy and Python's random generator.

---

## 4. The Tensor Object (`plast.Tensor`)

The `Tensor` class wraps the C++ implementation.

### Attributes
* **`shape`** (list): List representing the dimensions of the tensor.
* **`ndim`** (int): Number of dimensions (rank).
* **`device`** (plast.Device): The device containing the tensor data (`plast.Device.CPU` or `plast.Device.CUDA`).
* **`dtype`** (plast.DType): The scalar type (currently `plast.DType.Float32`).
* **`requires_grad`** (bool): If `True`, tracks operations to compute gradients during the backward pass.
* **`grad`** (plast.Tensor): Reference to the gradient tensor.
* **`creator`** (C Node): The computational graph node that created this tensor, or `None` if it is a leaf.
* **`strides`** (list): Layout strides in memory.
* **`is_contiguous`** (bool): Whether the memory layout is contiguous.
* **`T`** (plast.Tensor): Transposed 2-D matrix view. Only supported on 2-D tensors.

### Core Methods
* **`numel()`** $\rightarrow$ `int`: Returns the total number of elements in the tensor.
* **`numpy()`** $\rightarrow$ `np.ndarray`: Realizes the forward pass and returns a NumPy array view (shares CPU memory).
* **`copy_from_numpy(arr)`**: Overwrites the tensor's data buffer with a float32 NumPy array `arr` of matching shape.
* **`item()`** $\rightarrow$ `float`: Returns the value of a single-element tensor as a standard Python float.
* **`size(dim=None)`**: Returns the shape as a list, or the size of a specific dimension `dim` (supports negative indexing).
* **`clone()`** $\rightarrow$ `Tensor`: Returns a deep copy of the tensor with separate memory allocation.
* **`detach()`** $\rightarrow$ `Tensor`: Returns a new tensor sharing the same storage but detached from the autograd graph (with `requires_grad=False`).
* **`contiguous()`** $\rightarrow$ `Tensor`: Returns a contiguous copy of the tensor (returns `self` if already contiguous).
* **`to(device)`** / **`cpu()`** / **`cuda()`**: Transfers the tensor to the target device.
* **`forward()`** / **`realize()`**: Executes the scheduled DAG traversal to populate the tensor's data.
* **`backward()`**: Triggers backpropagation to calculate gradients for all ancestors. Only supported on scalar tensors (single element).

### Math & Operators
Plast supports standard arithmetic operators (`+`, `-`, `*`, `/`, `@` (matrix multiplication), `**` (exponentiation), negation, and absolute value) as well as named method counterparts:
* **`add(other)`** / **`sub(other)`** / **`mul(other)`** / **`div(other)`** / **`pow(exponent)`** / **`matmul(other)`**

#### Element-Wise Math Functions
* **`log()`** / **`exp()`** / **`sin()`** / **`cos()`** / **`tan()`** / **`abs()`** / **`sqrt()`**
* **`relu()`** / **`sigmoid()`** / **`tanh()`** / **`softmax(dim=-1)`**

#### Reductions
* **`sum(dim=None, keepdim=False)`**
* **`mean(dim=None, keepdim=False)`**
* **`min(dim=None, keepdim=False)`**
* **`max(dim=None, keepdim=False)`**
* **`norm(p=2, dim=None, keepdim=False)`**: Computes the vector $p$-norm.

### Shape Transformations (Zero-Copy Strides)
* **`view(*shape)`** / **`reshape(*shape)`**: Returns a view of the tensor with a new shape. A single dimension can be `-1` to be inferred.
* **`transpose(dim0, dim1)`**: Swaps dimensions `dim0` and `dim1`.
* **`permute(*dims)`**: Permutes the dimensions according to the given order.
* **`squeeze(dim=None)`**: Removes dimensions of size 1.
* **`unsqueeze(dim)`**: Inserts a dimension of size 1 at position `dim`.
* **`flatten(start_dim=0, end_dim=-1)`**: Flattens a range of dimensions into a single vector.
* **`expand(*shape)`**: Expands size-1 dimensions to match the target `shape` without copy.

### Graph Decorators
* **`@plast.jit`** / **`with plast.jit:`**: Caches graph structures on their first run. On subsequent calls, JIT bypasses graph construction, scheduling, and fusion checks.
* **`@plast.no_grad()`** / **`with plast.no_grad():`**: Context manager/decorator that disables autograd tracing to speed up evaluation or inference.

---

## 5. Neural Network Modules (`plast.nn`)

### Base Class: `plast.nn.Module`
The base class for all neural network modules. Tracks registered submodules and parameters.

#### Properties
* **`training`** (bool): Boolean state of training mode vs. evaluation mode.

#### Methods
* **`parameters()`**: Returns a flat list of all learnable parameter tensors (`plast.Parameter`).
* **`named_parameters(prefix="")`**: Returns an iterator yielding `(name, parameter)` pairs.
* **`num_parameters(only_trainable=True)`**: Returns the count of trainable scalar values.
* **`named_modules()`** / **`modules()`**: Returns iterators over registered modules.
* **`train(mode=True)`** / **`eval()`**: Sets the module and all sub-modules to training or evaluation mode (affects `Dropout` and `BatchNorm1d`).
* **`to(device)`** / **`cpu()`** / **`cuda()`**: Transfers all registered model parameters to the target device.
* **`zero_grad()`**: Resets the gradient buffers of all model parameters.
* **`state_dict(prefix="")`**: Returns a flat dict mapping parameter names to float32 NumPy arrays.
* **`load_state_dict(state_dict, prefix="", strict=True)`**: Restores parameters from a state dict.

### Containers
* **`plast.nn.Sequential(*layers)`**: Runs modules sequentially. Accepts a positional list of modules or a dictionary of name-module pairs.
* **`plast.nn.ModuleList(modules=None)`**: Holds a list of submodules, registering them correctly for parameter discovery.

### Standard Layers
* **`plast.nn.Linear(in_features, out_features, bias=True, device=Device.CPU)`**: A fully-connected linear layer. Weights are initialized using Kaiming normal initialization.
* **`plast.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, bias=True, device=Device.CPU)`**: 2-D Convolutional layer. Supports integer or tuple `kernel_size`.
* **`plast.nn.BatchNorm1d(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, device=Device.CPU)`**: Batch normalization layer for 2-D inputs `[N, C]` or 3-D inputs `[N, C, L]`.
* **`plast.nn.LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, device=Device.CPU)`**: Layer normalization layer.
* **`plast.nn.Dropout(p=0.5)`**: During training, randomly zeroes elements of the input tensor with probability `p`.

### Activations
All activations can be used as modules inside `nn.Sequential` or as standalone modules:
* **`plast.nn.ReLU()`**
* **`plast.nn.LeakyReLU(negative_slope=0.01)`**
* **`plast.nn.Sigmoid()`**
* **`plast.nn.Tanh()`**
* **`plast.nn.Softmax(dim=-1)`**
* **`plast.nn.GELU()`**
* **`plast.nn.SiLU()`**

### Loss Functions
* **`plast.nn.MSELoss(reduction='mean')`**: Mean Squared Error loss.
* **`plast.nn.L1Loss(reduction='mean')`**: Mean Absolute Error (L1) loss.
* **`plast.nn.SmoothL1Loss(beta=1.0, reduction='mean')`**: Smooth L1 (Huber) loss.
* **`plast.nn.CrossEntropyLoss(reduction='mean')`**: Softmax cross-entropy loss for classification. Supports target class indices `[N]` or one-hot targets `[N, C]`.
* **`plast.nn.NLLLoss(reduction='mean')`**: Negative log-likelihood loss. Expects input to be log-probabilities.
* **`plast.nn.BCELoss(reduction='mean')`**: Binary Cross Entropy loss. Expects inputs to be sigmoid probability outputs.
* **`plast.nn.BCEWithLogitsLoss(reduction='mean')`**: BCE loss applied to raw logits (numerically more stable).

---

## 6. Functional Interface (`plast.nn.functional`)

The functional interface contains raw mathematical transformations:
* **`relu(x)`** / **`leaky_relu(x, negative_slope=0.01)`** / **`sigmoid(x)`** / **`tanh(x)`** / **`gelu(x)`** / **`silu(x)`**
* **`softmax(x, dim=-1)`** / **`log_softmax(x, dim=-1)`**
* **`dropout(x, p=0.5, training=True)`**
* **`linear(input, weight, bias=None)`**
* **`conv2d(input, weight, bias=None, stride=1)`**
* **`batch_norm(input, running_mean, running_var, weight=None, bias=None, training=True, momentum=0.1, eps=1e-5)`**
* **`layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5)`**
* **`mse_loss(input, target, reduction='mean')`** / **`l1_loss(input, target, reduction='mean')`** / **`smooth_l1_loss(input, target, beta=1.0, reduction='mean')`**
* **`cross_entropy(input, target, reduction='mean')`** / **`nll_loss(input, target, reduction='mean')`**
* **`binary_cross_entropy(input, target, reduction='mean')`** / **`binary_cross_entropy_with_logits(input, target, reduction='mean')`**

---

## 7. Optimizers & Schedulers (`plast.optim`)

### Base Class: `plast.optim.Optimizer`
Manages optimization parameter groups.
* **`step()`**: Updates model parameters based on gradients.
* **`zero_grad()`**: Resets the gradients of all tracked parameters.
* **`state_dict()`** / **`load_state_dict(state_dict)`**: Saves/loads optimizer hyperparameters.

### Optimizers
* **`plast.optim.SGD(params, lr=0.01)`**: Stochastic Gradient Descent.
* **`plast.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8)`**: Adam optimizer.
* **`plast.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)`**: Adam with decoupled weight decay.
> [!NOTE]
> Adam and AdamW parameter updates are currently implemented only on the CPU. Move your parameters to CPU before running Adam/AdamW stepping.

### Learning Rate Schedulers
* **`StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)`**: Decays the learning rate of each parameter group by `gamma` every `step_size` epochs.
* **`MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)`**: Decays the learning rate of each parameter group by `gamma` when the epoch count reaches milestone marks.
* **`ExponentialLR(optimizer, gamma, last_epoch=-1)`**: Decays the learning rate exponentially by `gamma` every epoch.
* **`CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)`**: Applies cosine annealing.
* **`ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, ...)`**: Decays the learning rate when a tracked validation metric plateaus.

---

## 8. Data Loading Utilities (`plast.data`)

* **`plast.data.Dataset`**: Abstract base class for custom datasets. Subclasses must implement `__len__()` and `__getitem__(index)`.
* **`plast.data.TensorDataset(*tensors)`**: Wraps input and target arrays. Auto-converts inputs to persistent CPU tensors.
* **`plast.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, device=None)`**: Iterator yielding batched and shuffled tensors. Stages yielded batches directly onto the target execution hardware.

---

## 9. Experiment Tracking (`plast.experiment`)

### `ExperimentConfig`
Defines metadata, model configurations, and training parameters for a run.
```python
config = ExperimentConfig(
    name="mnist_mlp",
    model={"layers": [784, 128, 10], "activation": "relu"},
    training={"lr": 0.05, "epochs": 5, "batch_size": 64},
    device="cpu",
    seed=42,
    notes="Baseline SGD run"
)
```

### `ExperimentTracker`
Tracks parameters, updates metric tables, and automatically manages model checkpoints.
```python
tracker = ExperimentTracker(config, base_dir="./experiments", verbose=True)

for epoch in range(epochs):
    # training code
    tracker.log_epoch(epoch, {"train_loss": loss, "val_accuracy": acc}, model=model)

tracker.finish()
```
Logs are saved in the directory `experiments/{experiment_name}/run_NNN/`, generating:
* `config.yaml`: Frozen config details.
* `metrics.yaml`: Per-epoch metrics.
* `checkpoints/best_model.npz`: Saved weights of the best performing epoch automatically tracked using `val_accuracy` (or `val_loss`).
* `summary.yaml` (top level): A directory-wide leaderboard summarizing runs.
