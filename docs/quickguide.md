# quick guide

plast has two APIs. choose your fighter.

## the classic api (for noobs)

pytorch-compatible. `nn.Module`, `nn.Sequential`, optimizers, dataloaders. if you know torch, you know this.

```python
import plast
from plast.nn import Linear, ReLU, Sequential, MSELoss
from plast.optim import SGD

plast.init_arenas(device=plast.Device.CPU)

model = Sequential(
    Linear(2, 8),
    ReLU(),
    Linear(8, 1),
)
optimizer = SGD(model.parameters(), lr=0.01)
loss_fn = MSELoss()

x = plast.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
y = plast.tensor([[0], [1], [1], [0]])

for epoch in range(1000):
    pred = model(x)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## the pipeline api (for pros)

neural networks are data transformation pipelines. compose pure functions, let the scheduler and JIT handle the rest.

```python
import plast as p
from plast.experiment import ExperimentConfig, ExperimentTracker

p.init_arenas(device=p.Device.CUDA)

# define your pipeline as function composition
# (pipe() api coming soon — for now, chain manually)
def forward(x, W1, b1, W2, b2):
    h = p.nn.functional.relu(p.nn.functional.linear(x, W1, b1))
    return p.nn.functional.linear(h, W2, b2)

# tensors
x = p.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
W1, b1, W2, b2 = [p.tensor(...) for _ in range(4)]

loss = forward(x, W1, b1, W2, b2)
loss.backward()
```

## tensors

```python
import plast
plast.init_arenas(meta_size_mb=10, data_size_mb=100, device=plast.Device.CPU)

# from data
x = plast.tensor([[1, 2], [3, 4]], device=plast.Device.CPU)

# arithmetic — these all build graph nodes
z = x + y          # add
z = x * y          # mul
z = x @ y          # matmul

# move devices
z = z.to(plast.Device.CUDA)

# back to numpy
z_np = z.numpy()
```

## autograd

```python
import plast
plast.init_arenas(device=plast.Device.CPU)

x = plast.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = plast.tensor([4.0, 5.0, 6.0], requires_grad=True)
loss = ((x * y) ** 2).mean()

loss.backward()  # populates x.grad and y.grad
print(x.grad.numpy())
```

## gpu training

change `device=plast.Device.CUDA` in `init_arenas()` and when creating tensors. everything else stays the same — arena dispatch handles the rest.

## experiment tracking

```python
from plast.experiment import ExperimentConfig, ExperimentTracker

config = ExperimentConfig(
    name="mlp_mnist",
    model={"hidden": 256},
    training={"lr": 1e-3, "epochs": 50},
    device="cuda",
)
tracker = ExperimentTracker(config)

for epoch in range(50):
    loss = train_one_epoch(...)
    tracker.log_epoch(epoch, {"train_loss": loss})

tracker.finish()
# produces experiments/mlp_mnist/run_001/ with config, metrics, checkpoints
```

## c api

plast is a C library with python bindings. if you want to go bare metal:

```c
#include "arena.h"
#include "tensor.h"
#include "graph.h"
#include "op.h"

Arena meta = arena_create(Mib(10), DEVICE_CPU);
Arena data = arena_create(Mib(100), DEVICE_CUDA);

// 2D parameter tensor
Tensor *w = init(&meta, &data, DEVICE_CUDA, FLOAT32,
                 (u64[]){2, 8}, 2, true, rand_init);

// graph node
Tensor *out = init(&meta, &data, DEVICE_CUDA, FLOAT32,
                   (u64[]){4, 8}, 2, true, NULL);
Node *n = arena_node_alloc(&meta, (Tensor *[]){x, w}, 2,
                           out, get_op_impl(MATMUL), 0, false);

// forward/backward
forward(n);
set_ones_grad(out);
backward(n);
```

see `main.c` for a complete XOR training example.

## project structure

```
include/            — C headers
  kernels/             — kernel declarations
  optimizers/          — optimizer structs
src/                — C/CUDA implementation
  kernels/cpu/         — cpu kernels (avx, omp)
  kernels/cuda/        — cuda kernels (sm_80+)
  optimizers/          — sgd, adam, adamw
  python/              — pybind11 bindings
plast/              — python package
  nn/                  — classic api (module, layers, losses)
  optim/               — optimizers + lr schedulers
  data/                — dataset, dataloader
  experiment/          — tracking, config, yaml
tests/              — pytest suite
```

## key concepts

- **arena allocation** — pre-allocated pools. zero malloc/free during training.
- **dynamic computation graph** — built per forward/backward, topological sort.
- **dual backend** — cpu (avx/simd + omp) and cuda (sm_80+) share the same graph interface.
- **pipeline philosophy** — layer ops are pure functions. compose them. the JIT fuses them.

## see also

- `main.c` — native C XOR example
- `test_modular_xor.py` — full python example with experiment tracking
- `tests/` — the test suite documents expected behavior better than any readme
