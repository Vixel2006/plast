"""plast — a high-performance deep learning engine from scratch.

Quick start::

    import plast as p

    # tensor creation
    x = p.randn([32, 784])
    w = p.zeros([784, 256])

    # classic nn.Module API
    net = p.nn.Sequential(
        p.nn.Linear(784, 256),
        p.nn.ReLU(),
        p.nn.Linear(256, 10),
    )
    loss_fn = p.nn.CrossEntropyLoss()
    opt     = p.optim.Adam(net.parameters(), lr=1e-3)

    # training step
    with p.arena_scope():
        logits = net(x)
        loss   = loss_fn(logits, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
"""

from .plast_core import Device, DType, OpType, Pass, Scheduler
from .tensor import Tensor, Parameter, forward, backward, jit, no_grad
from ._internal import (
    init_arenas,
    tensor,
    get_arenas,
    get_persistent_arenas,
    reset_transient_arenas,
    arena_scope,
)

from . import nn
from . import optim
from . import data
from . import experiment

import numpy as np


# ── Tensor factory functions ──

def zeros(shape, *, device=Device.CPU, requires_grad=False) -> Tensor:
    """Return a tensor of zeros with the given *shape*.

    Example::

        b = plast.zeros([1, 256])
    """
    return tensor(np.zeros(shape, dtype=np.float32), device=device, requires_grad=requires_grad)


def ones(shape, *, device=Device.CPU, requires_grad=False) -> Tensor:
    """Return a tensor of ones with the given *shape*.

    Example::

        mask = plast.ones([batch, seq_len])
    """
    return tensor(np.ones(shape, dtype=np.float32), device=device, requires_grad=requires_grad)


def zeros_like(t: Tensor, *, requires_grad=False) -> Tensor:
    """Return a zero tensor with the same shape and device as *t*."""
    return zeros(t.shape, device=t.device, requires_grad=requires_grad)


def ones_like(t: Tensor, *, requires_grad=False) -> Tensor:
    """Return a ones tensor with the same shape and device as *t*."""
    return ones(t.shape, device=t.device, requires_grad=requires_grad)


def full(shape, fill_value, *, device=Device.CPU, requires_grad=False) -> Tensor:
    """Return a tensor filled with *fill_value*.

    Example::

        t = plast.full([3, 3], 5.0)
    """
    arr = np.full(shape, fill_value, dtype=np.float32)
    return tensor(arr, device=device, requires_grad=requires_grad)


def full_like(t: Tensor, fill_value, *, requires_grad=False) -> Tensor:
    """Return a tensor filled with *fill_value*, same shape/device as *t*."""
    return full(t.shape, fill_value, device=t.device, requires_grad=requires_grad)


def randn(shape, *, device=Device.CPU, requires_grad=False) -> Tensor:
    """Return a tensor filled with samples from N(0, 1).

    Example::

        x = plast.randn([batch_size, 784])
    """
    arr = np.random.randn(*shape).astype(np.float32)
    return tensor(arr, device=device, requires_grad=requires_grad)


def rand(shape, *, device=Device.CPU, requires_grad=False) -> Tensor:
    """Return a tensor filled with samples from Uniform(0, 1).

    Example::

        mask = plast.rand([batch, seq])
    """
    arr = np.random.rand(*shape).astype(np.float32)
    return tensor(arr, device=device, requires_grad=requires_grad)


def randint(low, high, shape, *, device=Device.CPU) -> Tensor:
    """Return integer samples from Uniform(low, high) as a float32 tensor.

    Example::

        labels = plast.randint(0, 10, [batch_size])
    """
    arr = np.random.randint(low, high, size=shape).astype(np.float32)
    return tensor(arr, device=device)


def eye(n, *, device=Device.CPU, requires_grad=False) -> Tensor:
    """Return the *n×n* identity matrix.

    Example::

        I = plast.eye(4)
    """
    arr = np.eye(n, dtype=np.float32)
    return tensor(arr, device=device, requires_grad=requires_grad)


def arange(start, stop=None, step=1, *, device=Device.CPU) -> Tensor:
    """Return evenly spaced values (like ``np.arange``).

    Example::

        t = plast.arange(10)       # [0, 1, ..., 9]
        t = plast.arange(2, 10, 2) # [2, 4, 6, 8]
    """
    if stop is None:
        start, stop = 0, start
    arr = np.arange(start, stop, step, dtype=np.float32)
    return tensor(arr, device=device)


def linspace(start, end, steps, *, device=Device.CPU) -> Tensor:
    """Return *steps* evenly spaced values from *start* to *end* inclusive.

    Example::

        t = plast.linspace(0.0, 1.0, 100)
    """
    arr = np.linspace(start, end, steps, dtype=np.float32)
    return tensor(arr, device=device)


# ── Higher-level tensor operations ──

def cat(tensors, dim=0) -> Tensor:
    """Concatenate a sequence of tensors along an existing dimension.

    Example::

        out = plast.cat([a, b, c], dim=0)
    """
    if not tensors:
        raise ValueError("cat() received an empty sequence of tensors.")
    arrays = [t.numpy() for t in tensors]
    result_np = np.concatenate(arrays, axis=dim).astype(np.float32)
    device = tensors[0].device
    return tensor(result_np, device=device)


def stack(tensors, dim=0) -> Tensor:
    """Stack a sequence of tensors along a **new** dimension.

    Example::

        batched = plast.stack([row1, row2, row3], dim=0)
    """
    if not tensors:
        raise ValueError("stack() received an empty sequence of tensors.")
    arrays = [t.numpy() for t in tensors]
    result_np = np.stack(arrays, axis=dim).astype(np.float32)
    device = tensors[0].device
    return tensor(result_np, device=device)


def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    """Return elements from *x* where *condition* is non-zero, else from *y*.

    Example::

        out = plast.where(mask, a, b)
    """
    cond_np = condition.numpy().astype(bool)
    x_np = x.numpy()
    y_np = y.numpy()
    result_np = np.where(cond_np, x_np, y_np).astype(np.float32)
    device = x.device
    return tensor(result_np, device=device)


def clip(t: Tensor, min_val=None, max_val=None) -> Tensor:
    """Clamp all elements of *t* into the range [*min_val*, *max_val*].

    Example::

        out = plast.clip(grads, -1.0, 1.0)
    """
    result_np = np.clip(t.numpy(), min_val, max_val).astype(np.float32)
    return tensor(result_np, device=t.device)


# convenience alias
clamp = clip


# ── Model persistence helpers ──

def save(model_or_state, path: str) -> None:
    """Save a model's weights (or any state dict) to a ``.npz`` file.

    Args:
        model_or_state: An :class:`~plast.nn.Module` or a ``dict`` mapping
                        parameter names to numpy arrays.
        path:           Destination path.  A ``.npz`` extension is added if
                        not already present.

    Example::

        plast.save(model, "checkpoints/epoch_10.npz")
    """
    if not path.endswith(".npz"):
        path = path + ".npz"
    if hasattr(model_or_state, "state_dict"):
        state = model_or_state.state_dict()
    else:
        state = dict(model_or_state)
    np.savez(path, **state)


def load(model, path: str, strict: bool = True) -> None:
    """Load weights from a ``.npz`` file into *model*.

    Args:
        model:  An :class:`~plast.nn.Module` to load weights into.
        path:   Path to the ``.npz`` checkpoint file.
        strict: If ``True`` (default), raise an error if any key in the
                checkpoint is missing from the model.

    Example::

        plast.load(model, "checkpoints/best.npz")
    """
    if not path.endswith(".npz"):
        path = path + ".npz"
    checkpoint = dict(np.load(path))
    expected = set(model.state_dict().keys())
    got = set(checkpoint.keys())
    if strict and (expected - got):
        missing = expected - got
        raise RuntimeError(
            f"load(): checkpoint is missing keys: {sorted(missing)}. "
            "Pass strict=False to ignore missing keys."
        )
    if got - expected:
        unexpected = got - expected
        import warnings
        warnings.warn(
            f"load(): checkpoint contains unexpected keys that will be ignored: "
            f"{sorted(unexpected)}"
        )
    model.load_state_dict(checkpoint)


# ── Gradient clipping utility ──

def clip_grad_norm_(parameters, max_norm: float, norm_type: float = 2.0) -> float:
    """Clip the gradient norm of an iterable of parameters **in-place**.

    This rescales all gradients jointly so the total norm does not exceed
    *max_norm*.  Returns the total norm before clipping.

    Example::

        total_norm = plast.clip_grad_norm_(model.parameters(), max_norm=1.0)
    """
    params = list(parameters)
    grads = []
    for p in params:
        if p.grad is not None:
            grads.append(p.grad)

    if not grads:
        return 0.0

    # Compute total norm
    if norm_type == float("inf"):
        total_norm = max(abs(g).max().item() for g in grads)
    else:
        total_norm = sum(
            float(np.sum(np.abs(g) ** norm_type)) for g in grads
        ) ** (1.0 / norm_type)

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params:
            if p.grad is not None:
                # gradients are raw C tensors — scale via numpy
                g_np = p.grad.numpy()
                p._t.copy_from_numpy((g_np * clip_coef).astype(np.float32))

    return float(total_norm)


# ── Seed helper ──

def manual_seed(seed: int) -> None:
    """Set the random seed for reproducibility (both numpy and Python random).

    Example::

        plast.manual_seed(42)
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
