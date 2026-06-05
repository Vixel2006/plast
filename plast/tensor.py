import numpy as np

from .plast_core import (
    Tensor as _CTensor,
    Pass,
    Scheduler,
    OpType,
    create_node,
    tensor_init,
    Device,
    DType,
    set_ones_grad,
    numel as _numel,
)
from ._internal import get_arenas

_builtin_max = max

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
            raise ValueError(f"Cannot broadcast shapes {shape1} and {shape2}")
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
    """Decorator that enables JIT caching in the scheduler.

    Use as a decorator:
        @plast.jit
        def train_step(loss):
            plast.forward(loss)
            loss.backward()

    When JIT is enabled, the scheduler caches the DAG on first execution
    and reuses it on subsequent calls within the same arena epoch.
    """

    def __call__(self, fn):
        import functools

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            _get_scheduler().set_jit_mode(True)
            try:
                return fn(*args, **kwargs)
            finally:
                _get_scheduler().set_jit_mode(False)

        return wrapper


jit = _Jit()


def forward(tensor):
    if tensor.creator:
        _get_scheduler().schedule(tensor.creator, Pass.FORWARD)


def backward(tensor):
    if tensor.creator:
        _get_scheduler().schedule(tensor.creator, Pass.BACKWARD)


class Tensor:
    _t: _CTensor

    def __init__(self, inner: _CTensor):
        if isinstance(inner, _CTensor):
            self._t = inner
        else:
            raise TypeError(f"Tensor() requires a C Tensor, got {type(inner)}")

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
    def requires_grad(self, val):
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

    # ── NumPy bridge ──

    def numel(self):
        return _numel(self._t)

    def numpy(self):
        return self._t.numpy()

    def copy_from_numpy(self, arr):
        self._t.copy_from_numpy(arr)

    def item(self):
        if np.prod(self.shape) != 1:
            raise ValueError(
                "only single-element tensors can be converted to Python scalars"
            )
        return float(self.numpy().flatten()[0])

    def size(self):
        return self.shape

    def __repr__(self):
        np_str = str(self.numpy())
        dev = ", device=CUDA" if self.device == Device.CUDA else ""
        grad = ", requires_grad=True" if self.requires_grad else ""
        return f"tensor({np_str}{dev}{grad})"

    def __str__(self):
        return self.__repr__()

    # ── Device ──

    def to(self, device):
        if self.device == device:
            return self
        from ._internal import get_arenas, get_persistent_arenas

        is_param = isinstance(self, Parameter)
        meta, data = get_persistent_arenas() if is_param else get_arenas()
        raw = tensor_init(
            meta, data, device, self.dtype, list(self.shape), self.requires_grad
        )
        raw.copy_from_numpy(self.numpy())
        t = Tensor.__new__(Parameter if is_param else Tensor)
        t._t = raw
        return t

    # ── Scheduler-aware forward/backward/realize ──

    def forward(self):
        if self.creator:
            _get_scheduler().schedule(self.creator, Pass.FORWARD)

    def backward(self):
        if len(self.shape) == 1 and self.shape[0] == 1:
            set_ones_grad(_unwrap(self))
        if self.creator:
            _get_scheduler().schedule(self.creator, Pass.BACKWARD)

    def realize(self):
        self.forward()

    # ── Arithmetic operators ──

    def __add__(self, other):
        other = _to_tensor(other, self.device)
        return _run_op(
            [self, other], OpType.ADD, broadcast_shape(self.shape, other.shape)
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = _to_tensor(other, self.device)
        return _run_op(
            [self, other], OpType.SUB, broadcast_shape(self.shape, other.shape)
        )

    def __rsub__(self, other):
        other = _to_tensor(other, self.device)
        return _run_op(
            [other, self], OpType.SUB, broadcast_shape(other.shape, self.shape)
        )

    def __mul__(self, other):
        other = _to_tensor(other, self.device)
        return _run_op(
            [self, other], OpType.MUL, broadcast_shape(self.shape, other.shape)
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = _to_tensor(other, self.device)
        return _run_op(
            [self, other], OpType.DIV, broadcast_shape(self.shape, other.shape)
        )

    def __rtruediv__(self, other):
        other = _to_tensor(other, self.device)
        return _run_op(
            [other, self], OpType.DIV, broadcast_shape(other.shape, self.shape)
        )

    def __neg__(self):
        return _run_op([self], OpType.NEG, list(self.shape))

    def __abs__(self):
        return _run_op([self], OpType.ABS, list(self.shape))

    def __matmul__(self, other):
        if len(self.shape) != 2 or len(other.shape) != 2:
            raise ValueError("matmul currently only supports 2D matrices")
        if self.shape[1] != other.shape[0]:
            raise ValueError(f"Dimension mismatch: {self.shape} @ {other.shape}")
        return _run_op([self, other], OpType.MATMUL, [self.shape[0], other.shape[1]])

    # ── Math functions ──

    def log(self):
        return _run_op([self], OpType.LOG, list(self.shape))

    def exp(self):
        return _run_op([self], OpType.EXP, list(self.shape))

    def sin(self):
        return _run_op([self], OpType.SIN, list(self.shape))

    def cos(self):
        return _run_op([self], OpType.COS, list(self.shape))

    def tan(self):
        return _run_op([self], OpType.TAN, list(self.shape))

    # ── Reductions ──

    def sum(self, dim=None, keepdim=False):
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

    def mean(self, dim=None, keepdim=False):
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

    def min(self, dim=None, keepdim=False):
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

    def max(self, dim=None, keepdim=False):
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

    # ── Shape operations ──

    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        shape = list(shape)
        total = int(np.prod(self.shape))
        minus_one = -1
        prod = 1
        for i, s in enumerate(shape):
            if s == -1:
                if minus_one != -1:
                    raise ValueError("Only one dimension can be -1")
                minus_one = i
            else:
                prod *= s
        if minus_one != -1:
            if total % prod != 0:
                raise ValueError(f"Cannot view shape {self.shape} as {shape}")
            shape[minus_one] = total // prod
        elif np.prod(shape) != total:
            raise ValueError(f"Cannot view shape {self.shape} as {shape}")
        return _run_op([self], OpType.VIEW, shape)

    def reshape(self, *shape):
        return self.view(*shape)

    def transpose(self, dim0, dim1):
        if dim0 < 0:
            dim0 += self.ndim
        if dim1 < 0:
            dim1 += self.ndim
        out_shape = list(self.shape)
        out_shape[dim0], out_shape[dim1] = out_shape[dim1], out_shape[dim0]
        return _run_op([self], OpType.TRANSPOSE, out_shape, dim=dim0, keepdim=dim1)

    def squeeze(self, dim=None):
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

    def unsqueeze(self, dim):
        if dim < 0:
            dim += self.ndim + 1
        out_shape = list(self.shape)
        out_shape.insert(dim, 1)
        return _run_op([self], OpType.UNSQUEEZE, out_shape, dim=dim)

    def flatten(self, start_dim=0, end_dim=-1):
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

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        shape = list(shape)
        if len(shape) < self.ndim:
            raise ValueError(f"Cannot expand to shape {shape} from {self.shape}")
        for i in range(1, self.ndim + 1):
            if self.shape[-i] != 1 and self.shape[-i] != shape[-i]:
                raise ValueError(f"Cannot expand shape {self.shape} to {shape}")
        return _run_op([self], OpType.EXPAND, shape)


class Parameter(Tensor):
    def __init__(
        self, data, *, device=Device.CPU, dtype=DType.Float32, requires_grad=True
    ):
        if isinstance(data, _CTensor):
            self._t = data
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
