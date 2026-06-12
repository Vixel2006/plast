import numpy as np

from .plast_core import (
    Tensor as _CTensor,
    Pass,
    Scheduler,
    OpType,
    create_node,
    execute_forward,
    tensor_init,
    Device,
    DType,
    set_ones_grad,
    numel as _numel,
)
from ._internal import get_arenas

_builtin_max = max

# ── no_grad context / decorator ──

_no_grad_active = False


class no_grad:
    """Disable gradient tracking inside a block or around a function.

    Can be used as a **context manager** or a **decorator**::

        # context manager
        with plast.no_grad():
            out = model(x)

        # decorator
        @plast.no_grad()
        def evaluate(model, x):
            return model(x)

    Inside a ``no_grad`` block all tensor operations execute immediately
    without building a computation graph.  No gradient tensors are allocated
    and ``backward()`` is a no-op.

    Use for model evaluation and inference.
    """

    def __init__(self):
        self._prev = None

    def __enter__(self):
        global _no_grad_active
        self._prev = _no_grad_active
        _no_grad_active = True
        return self

    def __exit__(self, *args):
        global _no_grad_active
        _no_grad_active = self._prev

    def __call__(self, fn):
        """Allow use as a decorator: ``@plast.no_grad()``."""
        import functools

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with no_grad():
                return fn(*args, **kwargs)

        return wrapper


# ── Helpers ──


def broadcast_shape(shape1, shape2):
    ndim1 = len(shape1)
    ndim2 = len(shape2)
    max_ndim = _builtin_max(ndim1, ndim2)
    out_shape = []
    for i in range(1, max_ndim + 1):
        dim1 = shape1[-i] if i <= ndim1 else 1
        dim2 = shape2[-i] if i <= ndim2 else 1
        if dim1 == dim2:
            out_shape.append(dim1)
        elif dim1 == 1:
            out_shape.append(dim2)
        elif dim2 == 1:
            out_shape.append(dim1)
        else:
            raise ValueError(
                f"Cannot broadcast shapes {shape1} and {shape2}: "
                f"dimension {-i} has size {dim1} vs {dim2}, which are incompatible."
            )
    return out_shape[::-1]


def _unwrap(t):
    return t._t if isinstance(t, Tensor) else t


def _unwrap_many(ts):
    return [_unwrap(t) for t in ts]


def _to_tensor(val, device):
    if isinstance(val, Tensor):
        return val
    if isinstance(val, _CTensor):
        return Tensor(val)
    from ._internal import tensor as _make_tensor

    return _make_tensor(val, device=device)


def _run_op(inputs, op_type, out_shape, dim=0, keepdim=0, fval=0.0, requires_grad=None):
    meta, data = get_arenas()
    raw_inputs = _unwrap_many(inputs)
    device = raw_inputs[0].device

    if _no_grad_active:
        raw_out = tensor_init(meta, data, device, DType.Float32, out_shape, False)
        execute_forward(op_type, raw_inputs, raw_out, dim, keepdim, fval, meta)
        return Tensor(raw_out)

    if requires_grad is None:
        requires_grad = any(t.requires_grad for t in raw_inputs)
    raw_out = tensor_init(meta, data, device, DType.Float32, out_shape, requires_grad)
    create_node(meta, raw_inputs, raw_out, op_type, dim, keepdim, fval)
    return Tensor(raw_out)


# ── Scheduler ──
_scheduler = None


def _get_scheduler():
    global _scheduler
    if _scheduler is None:
        _scheduler = Scheduler(capacity=16)
    return _scheduler


class _Jit:
    """Decorator *and* context manager that enables JIT compilation.

    Use as a decorator::

        @plast.jit
        def train_step(loss):
            loss.backward()

    Or inline::

        with plast.jit:
            loss.backward()

    When JIT is enabled the computation graph is cached after its first
    execution and reused on subsequent calls, skipping DAG construction
    and fusion optimisation.
    """

    def __call__(self, fn):
        import functools

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with self:
                return fn(*args, **kwargs)

        return wrapper

    def __enter__(self):
        _get_scheduler().set_jit_mode(True)

    def __exit__(self, *args):
        _get_scheduler().set_jit_mode(False)


jit = _Jit()


def forward(tensor):
    tensor.forward()


def backward(tensor):
    tensor.backward()


# ── Tensor repr helpers ──

def _format_numpy_for_repr(arr):
    """Format a numpy array the same way PyTorch does: no dtype suffix, compact."""
    # Use numpy's default repr but strip the 'array(' wrapper
    old_opts = np.get_printoptions()
    np.set_printoptions(precision=4, floatmode="maxprec_equal", sign=" ", linewidth=80)
    try:
        s = repr(arr)
        # strip 'array(' prefix and trailing ')'
        if s.startswith("array("):
            s = s[6:]
            if s.endswith(", dtype=float32)"):
                s = s[: -len(", dtype=float32)")]
            elif s.endswith(")"):
                s = s[:-1]
    finally:
        np.set_printoptions(**old_opts)
    return s


class Tensor:
    _t: _CTensor
    _realized: bool = False

    def __init__(self, inner: _CTensor):
        if isinstance(inner, _CTensor):
            self._t = inner
            self._realized = False
        else:
            raise TypeError(
                f"Tensor.__init__ expects a C-level Tensor, got {type(inner).__name__}. "
                "Use plast.tensor() to create tensors from Python data."
            )

    # ── Properties proxied to C++ tensor ──

    @property
    def shape(self):
        return list(self._t.shape)

    @property
    def ndim(self):
        return self._t.ndim

    @property
    def device(self):
        return self._t.device

    @property
    def dtype(self):
        return self._t.dtype

    @property
    def requires_grad(self):
        return self._t.requires_grad

    @requires_grad.setter
    def requires_grad(self, val: bool):
        self._t.requires_grad = val

    @property
    def grad(self):
        return self._t.grad

    @property
    def creator(self):
        return self._t.creator

    @property
    def strides(self):
        return list(self._t.strides)

    @property
    def is_contiguous(self):
        return self._t.is_contiguous

    # ── Convenience shape aliases ──

    @property
    def T(self):
        """Transpose a 2-D tensor (shorthand for ``t.transpose(0, 1)``)."""
        if self.ndim != 2:
            raise ValueError(
                f"Tensor.T is only defined for 2-D tensors, but this tensor has {self.ndim} dimensions. "
                "Use .transpose(dim0, dim1) for higher-dimensional transpositions."
            )
        return self.transpose(0, 1)

    # ── NumPy bridge ──

    def numel(self) -> int:
        """Return the total number of elements in this tensor."""
        return _numel(self._t)

    def numpy(self) -> np.ndarray:
        """Return the underlying data as a NumPy array (no copy if on CPU)."""
        self.forward()
        return self._t.numpy()

    def copy_from_numpy(self, arr: np.ndarray) -> None:
        """Overwrite this tensor's data with *arr* (must match shape and dtype)."""
        arr = np.asarray(arr, dtype=np.float32)
        if list(arr.shape) != self.shape:
            raise ValueError(
                f"Shape mismatch: tensor has shape {self.shape} "
                f"but numpy array has shape {list(arr.shape)}."
            )
        self._t.copy_from_numpy(arr)

    def item(self) -> float:
        """Return this tensor's value as a Python float.

        The tensor must contain exactly one element.
        """
        n = self.numel()
        if n != 1:
            raise ValueError(
                f"item() can only be called on a tensor with a single element, "
                f"but this tensor has shape {self.shape} ({n} elements). "
                "Use .numpy() to retrieve all values."
            )
        return float(self.numpy().flat[0])

    def size(self, dim: int = None):
        """Return shape, or the size of a specific *dim*.

        Args:
            dim: If given, return the size along this dimension (negative indexing supported).
                 If ``None`` (default), return the full shape as a list.
        """
        if dim is None:
            return self.shape
        ndim = self.ndim
        if not (-ndim <= dim < ndim):
            raise IndexError(
                f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], "
                f"but got {dim})."
            )
        return self.shape[dim % ndim]

    def clone(self) -> "Tensor":
        """Return a deep copy of this tensor (shares no memory).

        The clone is added to the computation graph exactly like any other
        operation, so gradients flow through it.
        """
        from ._internal import tensor as _make_tensor

        t = _make_tensor(
            self.numpy().copy(),
            device=self.device,
            requires_grad=self.requires_grad,
        )
        return t

    def detach(self) -> "Tensor":
        """Return a new tensor that shares storage but is detached from the graph.

        The returned tensor has ``requires_grad=False`` and no creator node.
        """
        meta, data = get_arenas()
        raw = tensor_init(meta, data, self.device, self.dtype, list(self.shape), False)
        raw.copy_from_numpy(self.numpy())
        return Tensor(raw)

    def contiguous(self) -> "Tensor":
        """Return a contiguous tensor.  Returns *self* if already contiguous."""
        if self.is_contiguous:
            return self
        return self.clone()

    # ── Python scalar conversions ──

    def __float__(self) -> float:
        return self.item()

    def __int__(self) -> int:
        return int(self.item())

    def __bool__(self) -> bool:
        return bool(self.item() != 0.0)

    # ── Repr ──

    def __repr__(self) -> str:
        data = self.numpy()
        data_str = _format_numpy_for_repr(data)
        extras = []
        if self.device == Device.CUDA:
            extras.append("device='cuda'")
        if self.requires_grad:
            extras.append("requires_grad=True")
        suffix = ", " + ", ".join(extras) if extras else ""
        return f"tensor({data_str}{suffix})"

    def __str__(self) -> str:
        return self.__repr__()

    # ── Device ──

    def to(self, device) -> "Tensor":
        """Move this tensor to *device* (``Device.CPU`` or ``Device.CUDA``).

        Returns *self* if already on the requested device.
        """
        if self.device == device:
            return self
        from ._internal import get_arenas, get_persistent_arenas

        is_param = isinstance(self, Parameter)
        meta, data = get_persistent_arenas() if is_param else get_arenas()
        raw = tensor_init(meta, data, device, self.dtype, list(self.shape), self.requires_grad)
        raw.copy_from_numpy(self.numpy())
        t = Tensor.__new__(Parameter if is_param else Tensor)
        t._t = raw
        return t

    def cpu(self) -> "Tensor":
        """Move to CPU (no-op if already on CPU)."""
        return self.to(Device.CPU)

    def cuda(self) -> "Tensor":
        """Move to CUDA (no-op if already on CUDA)."""
        return self.to(Device.CUDA)

    # ── Scheduler-aware forward/backward/realize ──

    def forward(self) -> None:
        if _no_grad_active:
            return
        if self._realized:
            return
        if self.creator:
            meta, _ = get_arenas()
            _get_scheduler().schedule(self.creator, Pass.FORWARD, meta)
        self._realized = True

    def backward(self) -> None:
        """Compute gradients for this tensor via backpropagation.

        The tensor must be a scalar (shape ``[1]`` or single element).
        """
        if _no_grad_active:
            return
        if self.numel() != 1:
            raise RuntimeError(
                f"backward() can only be called on scalar tensors (single element), "
                f"but this tensor has shape {self.shape}. "
                "Call .sum() or .mean() first to reduce to a scalar."
            )
        self.forward()
        set_ones_grad(_unwrap(self))
        if self.creator:
            meta, _ = get_arenas()
            _get_scheduler().schedule(self.creator, Pass.BACKWARD, meta)

    def realize(self) -> None:
        """Alias for :meth:`forward`."""
        self.forward()

    # ── Arithmetic operators ──

    def __add__(self, other):
        other = _to_tensor(other, self.device)
        return _run_op([self, other], OpType.ADD, broadcast_shape(self.shape, other.shape))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = _to_tensor(other, self.device)
        return _run_op([self, other], OpType.SUB, broadcast_shape(self.shape, other.shape))

    def __rsub__(self, other):
        other = _to_tensor(other, self.device)
        return _run_op([other, self], OpType.SUB, broadcast_shape(other.shape, self.shape))

    def __mul__(self, other):
        other = _to_tensor(other, self.device)
        return _run_op([self, other], OpType.MUL, broadcast_shape(self.shape, other.shape))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = _to_tensor(other, self.device)
        return _run_op([self, other], OpType.DIV, broadcast_shape(self.shape, other.shape))

    def __rtruediv__(self, other):
        other = _to_tensor(other, self.device)
        return _run_op([other, self], OpType.DIV, broadcast_shape(other.shape, self.shape))

    def __neg__(self):
        return _run_op([self], OpType.NEG, list(self.shape))

    def __abs__(self):
        return _run_op([self], OpType.ABS, list(self.shape))

    def __pow__(self, exponent):
        """Element-wise power: ``x ** n``.

        Currently only integer and float scalar exponents are supported.
        """
        if not isinstance(exponent, (int, float)):
            raise TypeError(
                f"Unsupported exponent type: {type(exponent).__name__}. "
                "Only scalar int/float exponents are supported."
            )
        # x**n = exp(n * log(x)) — works for positive x
        return (self.log() * float(exponent)).exp()

    def __matmul__(self, other):
        if len(self.shape) != 2 or len(other.shape) != 2:
            raise ValueError(
                f"matmul currently only supports 2-D matrices, "
                f"got shapes {self.shape} @ {other.shape}."
            )
        if self.shape[1] != other.shape[0]:
            raise ValueError(
                f"matmul: inner dimensions must match, "
                f"but got {self.shape} @ {other.shape} "
                f"({self.shape[1]} ≠ {other.shape[0]})."
            )
        return _run_op([self, other], OpType.MATMUL, [self.shape[0], other.shape[1]])

    # ── Named method aliases for arithmetic ──

    def add(self, other) -> "Tensor":
        """Alias for ``self + other``."""
        return self.__add__(other)

    def sub(self, other) -> "Tensor":
        """Alias for ``self - other``."""
        return self.__sub__(other)

    def mul(self, other) -> "Tensor":
        """Alias for ``self * other``."""
        return self.__mul__(other)

    def div(self, other) -> "Tensor":
        """Alias for ``self / other``."""
        return self.__truediv__(other)

    def rdiv(self, other) -> "Tensor":
        """Alias for ``other / self``."""
        return self.__rtruediv__(other)

    def matmul(self, other) -> "Tensor":
        """Alias for ``self @ other``."""
        return self.__matmul__(other)

    def pow(self, exponent) -> "Tensor":
        """Alias for ``self ** exponent``."""
        return self.__pow__(exponent)

    # ── Math functions ──

    def log(self) -> "Tensor":
        """Element-wise natural logarithm."""
        return _run_op([self], OpType.LOG, list(self.shape))

    def exp(self) -> "Tensor":
        """Element-wise exponential."""
        return _run_op([self], OpType.EXP, list(self.shape))

    def sin(self) -> "Tensor":
        """Element-wise sine."""
        return _run_op([self], OpType.SIN, list(self.shape))

    def cos(self) -> "Tensor":
        """Element-wise cosine."""
        return _run_op([self], OpType.COS, list(self.shape))

    def tan(self) -> "Tensor":
        """Element-wise tangent."""
        return _run_op([self], OpType.TAN, list(self.shape))

    def abs(self) -> "Tensor":
        """Element-wise absolute value (alias for ``abs(t)``)."""
        return self.__abs__()

    def sqrt(self) -> "Tensor":
        """Element-wise square root (equivalent to ``t ** 0.5``)."""
        return (self.log() * 0.5).exp()

    def relu(self) -> "Tensor":
        """Apply ReLU activation in-place style (convenience method)."""
        from .nn import functional as F
        return F.relu(self)

    def sigmoid(self) -> "Tensor":
        """Apply sigmoid activation (convenience method)."""
        from .nn import functional as F
        return F.sigmoid(self)

    def tanh(self) -> "Tensor":
        """Apply tanh activation (convenience method)."""
        from .nn import functional as F
        return F.tanh(self)

    def softmax(self, dim: int = -1) -> "Tensor":
        """Apply softmax along *dim* (convenience method)."""
        from .nn import functional as F
        return F.softmax(self, dim=dim)

    # ── Reductions ──

    def sum(self, dim=None, keepdim=False) -> "Tensor":
        """Sum elements, optionally along *dim*."""
        k = 1 if keepdim else 0
        if dim is None:
            return _run_op([self], OpType.SUM, [1], dim=9, keepdim=k)
        if dim < 0:
            dim += self.ndim
        out_shape = list(self.shape)
        if keepdim:
            out_shape[dim] = 1
        else:
            out_shape.pop(dim)
            if not out_shape:
                out_shape = [1]
        return _run_op([self], OpType.SUM, out_shape, dim=dim, keepdim=k)

    def mean(self, dim=None, keepdim=False) -> "Tensor":
        """Compute mean, optionally along *dim*."""
        k = 1 if keepdim else 0
        if dim is None:
            return _run_op([self], OpType.MEAN, [1], dim=9, keepdim=k)
        if dim < 0:
            dim += self.ndim
        out_shape = list(self.shape)
        if keepdim:
            out_shape[dim] = 1
        else:
            out_shape.pop(dim)
            if not out_shape:
                out_shape = [1]
        return _run_op([self], OpType.MEAN, out_shape, dim=dim, keepdim=k)

    def min(self, dim=None, keepdim=False) -> "Tensor":
        """Return minimum, optionally along *dim*."""
        k = 1 if keepdim else 0
        if dim is None:
            return _run_op([self], OpType.MIN, [1], dim=9, keepdim=k)
        if dim < 0:
            dim += self.ndim
        out_shape = list(self.shape)
        if keepdim:
            out_shape[dim] = 1
        else:
            out_shape.pop(dim)
            if not out_shape:
                out_shape = [1]
        return _run_op([self], OpType.MIN, out_shape, dim=dim, keepdim=k)

    def max(self, dim=None, keepdim=False) -> "Tensor":
        """Return maximum, optionally along *dim*."""
        k = 1 if keepdim else 0
        if dim is None:
            return _run_op([self], OpType.MAX, [1], dim=9, keepdim=k)
        if dim < 0:
            dim += self.ndim
        out_shape = list(self.shape)
        if keepdim:
            out_shape[dim] = 1
        else:
            out_shape.pop(dim)
            if not out_shape:
                out_shape = [1]
        return _run_op([self], OpType.MAX, out_shape, dim=dim, keepdim=k)

    def norm(self, p=2, dim=None, keepdim=False) -> "Tensor":
        """Compute the *p*-norm.

        Args:
            p:       The order of the norm (default 2, i.e. Frobenius / L2).
            dim:     If given, reduce along this dimension.
            keepdim: If ``True``, retain the reduced dimension as size 1.
        """
        if p == 2:
            return (self * self).sum(dim=dim, keepdim=keepdim).sqrt()
        elif p == 1:
            return self.abs().sum(dim=dim, keepdim=keepdim)
        else:
            return (self.abs() ** p).sum(dim=dim, keepdim=keepdim) ** (1.0 / p)

    # ── Shape operations ──

    def view(self, *shape) -> "Tensor":
        """Return a tensor with the same data viewed as *shape*.

        A single ``-1`` dimension is automatically inferred.
        """
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        shape = list(shape)
        total = self.numel()
        minus_one = -1
        prod = 1
        for i, s in enumerate(shape):
            if s == -1:
                if minus_one != -1:
                    raise ValueError("Only one dimension can be inferred (-1) at a time.")
                minus_one = i
            else:
                prod *= s
        if minus_one != -1:
            if total % prod != 0:
                raise ValueError(
                    f"Cannot view tensor of shape {self.shape} ({total} elements) "
                    f"as {shape}: {total} is not divisible by {prod}."
                )
            shape[minus_one] = total // prod
        elif int(np.prod(shape)) != total:
            raise ValueError(
                f"Cannot view tensor of shape {self.shape} ({total} elements) "
                f"as {shape} ({int(np.prod(shape))} elements): element count mismatch."
            )
        return _run_op([self], OpType.VIEW, shape)

    def reshape(self, *shape) -> "Tensor":
        """Alias for :meth:`view`."""
        return self.view(*shape)

    def transpose(self, dim0: int, dim1: int) -> "Tensor":
        """Swap dimensions *dim0* and *dim1*."""
        ndim = self.ndim
        if dim0 < 0:
            dim0 += ndim
        if dim1 < 0:
            dim1 += ndim
        for d, name in ((dim0, "dim0"), (dim1, "dim1")):
            if not (0 <= d < ndim):
                raise IndexError(
                    f"transpose: {name}={d} is out of range for a {ndim}-D tensor."
                )
        out_shape = list(self.shape)
        out_shape[dim0], out_shape[dim1] = out_shape[dim1], out_shape[dim0]
        return _run_op([self], OpType.TRANSPOSE, out_shape, dim=dim0, keepdim=dim1)

    def permute(self, *dims) -> "Tensor":
        """Permute dimensions.  Like ``transpose`` but for arbitrary reorderings.

        Example::

            t = plast.randn([2, 3, 4])
            t.permute(2, 0, 1)  # → shape [4, 2, 3]
        """
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            dims = dims[0]
        dims = list(dims)
        ndim = self.ndim
        if len(dims) != ndim:
            raise ValueError(
                f"permute expects exactly {ndim} dimension indices, got {len(dims)}."
            )
        # Implement as a sequence of transposes
        result = self
        # Build target by iteratively placing each axis into position
        # (simple O(n^2) approach — tensors are small-rank in practice)
        current = list(range(ndim))  # tracks where each original dim currently is
        for target_pos, orig_dim in enumerate(dims):
            current_pos = current.index(orig_dim)
            if current_pos != target_pos:
                result = result.transpose(current_pos, target_pos)
                current[current_pos], current[target_pos] = current[target_pos], current[current_pos]
        return result

    def squeeze(self, dim=None) -> "Tensor":
        """Remove size-1 dimensions."""
        if dim is None:
            out_shape = [s for s in self.shape if s != 1]
            if not out_shape:
                out_shape = [1]
            return _run_op([self], OpType.SQUEEZE, out_shape, dim=8)
        if dim < 0:
            dim += self.ndim
        if self.shape[dim] != 1:
            return self
        out_shape = list(self.shape)
        out_shape.pop(dim)
        return _run_op([self], OpType.SQUEEZE, out_shape, dim=dim)

    def unsqueeze(self, dim: int) -> "Tensor":
        """Insert a new size-1 dimension at *dim*."""
        if dim < 0:
            dim += self.ndim + 1
        out_shape = list(self.shape)
        out_shape.insert(dim, 1)
        return _run_op([self], OpType.UNSQUEEZE, out_shape, dim=dim)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "Tensor":
        """Flatten dimensions from *start_dim* to *end_dim* (inclusive)."""
        if end_dim < 0:
            end_dim += self.ndim
        if start_dim < 0:
            start_dim += self.ndim
        out_shape = []
        for i in range(start_dim):
            out_shape.append(self.shape[i])
        flat = 1
        for i in range(start_dim, end_dim + 1):
            flat *= self.shape[i]
        out_shape.append(flat)
        for i in range(end_dim + 1, self.ndim):
            out_shape.append(self.shape[i])
        return _run_op([self], OpType.FLATTEN, out_shape)

    def expand(self, *shape) -> "Tensor":
        """Expand size-1 dimensions to match *shape* (no data copy)."""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        shape = list(shape)
        if len(shape) < self.ndim:
            raise ValueError(
                f"expand: the number of target dimensions ({len(shape)}) must be ≥ "
                f"the tensor's rank ({self.ndim})."
            )
        for i in range(1, self.ndim + 1):
            if self.shape[-i] != 1 and self.shape[-i] != shape[-i]:
                raise ValueError(
                    f"expand: cannot expand dimension {-i} from size {self.shape[-i]} "
                    f"to {shape[-i]} (only size-1 dimensions can be expanded)."
                )
        return _run_op([self], OpType.EXPAND, shape)

    # ── Iteration / indexing helpers ──

    def __len__(self) -> int:
        """Return the size of the first dimension (like NumPy / PyTorch)."""
        if self.ndim == 0:
            raise TypeError("len() of a 0-dimensional tensor")
        return self.shape[0]

    def __iter__(self):
        """Iterate over the first dimension, yielding slices as tensors."""
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        """Basic indexing: integer index along the first dimension.

        Only single-integer indexing is supported for now.
        """
        if not isinstance(idx, int):
            raise TypeError(
                f"Only integer indexing is supported, got {type(idx).__name__}."
            )
        n = self.shape[0]
        if idx < 0:
            idx += n
        if not (0 <= idx < n):
            raise IndexError(
                f"Index {idx} is out of bounds for dimension 0 with size {n}."
            )
        # Slice the numpy array and return a new tensor
        from ._internal import tensor as _make_tensor

        return _make_tensor(self.numpy()[idx], device=self.device)


class Parameter(Tensor):
    """A :class:`Tensor` that is always a learnable parameter.

    Parameters are stored in persistent arenas (they survive
    ``reset_transient_arenas()`` calls) and default to
    ``requires_grad=True``.

    Example::

        w = plast.Parameter(np.random.randn(128, 64))
        # equivalent to nn.Module registering it automatically
    """

    def __init__(self, data, *, device=Device.CPU, dtype=DType.Float32, requires_grad=True):
        if isinstance(data, _CTensor):
            self._t = data
        elif isinstance(data, Tensor):
            # Allow wrapping an existing Tensor
            self._t = data._t
        else:
            from ._internal import tensor as _make_tensor

            t = _make_tensor(
                data,
                device=device,
                dtype=dtype,
                requires_grad=requires_grad,
                persistent=True,
            )
            self._t = t._t

    def __repr__(self) -> str:
        data = self.numpy()
        data_str = _format_numpy_for_repr(data)
        extras = ["requires_grad=True"]
        if self.device == Device.CUDA:
            extras.insert(0, "device='cuda'")
        suffix = ", " + ", ".join(extras)
        return f"Parameter containing:\ntensor({data_str}{suffix})"
