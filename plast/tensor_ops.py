import numpy as np
from .plast_core import Tensor, OpType, create_node, tensor_init, Device, DType, forward, backward, set_ones_grad
from ._internal import get_arenas, tensor

_builtin_max = max

def _to_tensor(val, device):
    if isinstance(val, Tensor):
        return val
    return tensor(val, device=device)

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

def _run_op(inputs, op_type, out_shape, dim=0, keepdim=0, fval=0.0, requires_grad=None):
    meta, data = get_arenas()
    device = inputs[0].device
    if requires_grad is None:
        requires_grad = any(t.requires_grad for t in inputs)
    out = tensor_init(meta, data, device, DType.Float32, out_shape, requires_grad)
    create_node(meta, inputs, out, op_type, dim, keepdim, fval)
    return out

# Element-wise operations
def add(self, other):
    other = _to_tensor(other, self.device)
    out_shape = broadcast_shape(self.shape, other.shape)
    return _run_op([self, other], OpType.ADD, out_shape)

def sub(self, other):
    other = _to_tensor(other, self.device)
    out_shape = broadcast_shape(self.shape, other.shape)
    return _run_op([self, other], OpType.SUB, out_shape)

def rsub(self, other):
    other = _to_tensor(other, self.device)
    out_shape = broadcast_shape(other.shape, self.shape)
    return _run_op([other, self], OpType.SUB, out_shape)

def mul(self, other):
    other = _to_tensor(other, self.device)
    out_shape = broadcast_shape(self.shape, other.shape)
    return _run_op([self, other], OpType.MUL, out_shape)

def div(self, other):
    other = _to_tensor(other, self.device)
    out_shape = broadcast_shape(self.shape, other.shape)
    return _run_op([self, other], OpType.DIV, out_shape)

def rdiv(self, other):
    other = _to_tensor(other, self.device)
    out_shape = broadcast_shape(other.shape, self.shape)
    return _run_op([other, self], OpType.DIV, out_shape)

def neg(self):
    return _run_op([self], OpType.NEG, list(self.shape))

def abs_op(self):
    return _run_op([self], OpType.ABS, list(self.shape))

# Math functions
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

# Matmul
def matmul(self, other):
    if len(self.shape) != 2 or len(other.shape) != 2:
        raise ValueError("matmul currently only supports 2D matrices in Plast")
    if self.shape[1] != other.shape[0]:
        raise ValueError(f"Matrix dimension mismatch: {self.shape} and {other.shape}")
    out_shape = [self.shape[0], other.shape[1]]
    return _run_op([self, other], OpType.MATMUL, out_shape)

# Reductions
def sum(self, dim=None, keepdim=False):
    keepdim_val = 1 if keepdim else 0
    if dim is None:
        out_shape = [1]
        return _run_op([self], OpType.SUM, out_shape, dim=9, keepdim=keepdim_val)
    else:
        if dim < 0:
            dim += self.ndim
        out_shape = list(self.shape)
        if keepdim:
            out_shape[dim] = 1
        else:
            out_shape.pop(dim)
            if not out_shape:
                out_shape = [1]
        return _run_op([self], OpType.SUM, out_shape, dim=dim, keepdim=keepdim_val)

def mean(self, dim=None, keepdim=False):
    keepdim_val = 1 if keepdim else 0
    if dim is None:
        out_shape = [1]
        return _run_op([self], OpType.MEAN, out_shape, dim=9, keepdim=keepdim_val)
    else:
        if dim < 0:
            dim += self.ndim
        out_shape = list(self.shape)
        if keepdim:
            out_shape[dim] = 1
        else:
            out_shape.pop(dim)
            if not out_shape:
                out_shape = [1]
        return _run_op([self], OpType.MEAN, out_shape, dim=dim, keepdim=keepdim_val)

def min(self, dim=None, keepdim=False):
    keepdim_val = 1 if keepdim else 0
    if dim is None:
        out_shape = [1]
        return _run_op([self], OpType.MIN, out_shape, dim=9, keepdim=keepdim_val)
    else:
        if dim < 0:
            dim += self.ndim
        out_shape = list(self.shape)
        if keepdim:
            out_shape[dim] = 1
        else:
            out_shape.pop(dim)
            if not out_shape:
                out_shape = [1]
        return _run_op([self], OpType.MIN, out_shape, dim=dim, keepdim=keepdim_val)

def max(self, dim=None, keepdim=False):
    keepdim_val = 1 if keepdim else 0
    if dim is None:
        out_shape = [1]
        return _run_op([self], OpType.MAX, out_shape, dim=9, keepdim=keepdim_val)
    else:
        if dim < 0:
            dim += self.ndim
        out_shape = list(self.shape)
        if keepdim:
            out_shape[dim] = 1
        else:
            out_shape.pop(dim)
            if not out_shape:
                out_shape = [1]
        return _run_op([self], OpType.MAX, out_shape, dim=dim, keepdim=keepdim_val)

# Shape operations
def view(self, *shape):
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
    shape = list(shape)
    
    total_elements = int(np.prod(self.shape))
    minus_one_idx = -1
    current_prod = 1
    for i, s in enumerate(shape):
        if s == -1:
            if minus_one_idx != -1:
                raise ValueError("Only one dimension can be -1")
            minus_one_idx = i
        else:
            current_prod *= s
            
    if minus_one_idx != -1:
        if total_elements % current_prod != 0:
            raise ValueError(f"Cannot view shape {self.shape} as {shape}")
        shape[minus_one_idx] = total_elements // current_prod
    elif np.prod(shape) != total_elements:
        raise ValueError(f"Cannot view shape {self.shape} as {shape}")
        
    return _run_op([self], OpType.VIEW, shape)

def reshape(self, *shape):
    return self.view(*shape)

def transpose(self, dim0, dim1):
    if dim0 < 0: dim0 += self.ndim
    if dim1 < 0: dim1 += self.ndim
    out_shape = list(self.shape)
    out_shape[dim0], out_shape[dim1] = out_shape[dim1], out_shape[dim0]
    # We pass dim0 and dim1. In our C-node structure we updated keepdim to be u64!
    # So dim=dim0, keepdim=dim1.
    return _run_op([self], OpType.TRANSPOSE, out_shape, dim=dim0, keepdim=dim1)

def squeeze(self, dim=None):
    if dim is None:
        out_shape = [s for s in self.shape if s != 1]
        if not out_shape:
            out_shape = [1]
        # In C, squeeze takes the target dim, or passes MAX_NDIM if all.
        # Let's check squeeze signature or just pass dim=MAX_NDIM (which is 8).
        return _run_op([self], OpType.SQUEEZE, out_shape, dim=8)
    else:
        if dim < 0: dim += self.ndim
        if self.shape[dim] != 1:
            return self
        out_shape = list(self.shape)
        out_shape.pop(dim)
        return _run_op([self], OpType.SQUEEZE, out_shape, dim=dim)

def unsqueeze(self, dim):
    if dim < 0: dim += self.ndim + 1
    out_shape = list(self.shape)
    out_shape.insert(dim, 1)
    return _run_op([self], OpType.UNSQUEEZE, out_shape, dim=dim)

def flatten(self, start_dim=0, end_dim=-1):
    if end_dim < 0: end_dim += self.ndim
    if start_dim < 0: start_dim += self.ndim
    
    out_shape = []
    for i in range(start_dim):
        out_shape.append(self.shape[i])
        
    flat_size = 1
    for i in range(start_dim, end_dim + 1):
        flat_size *= self.shape[i]
    out_shape.append(flat_size)
    
    for i in range(end_dim + 1, self.ndim):
        out_shape.append(self.shape[i])
        
    return _run_op([self], OpType.FLATTEN, out_shape)

def expand(self, *shape):
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
    shape = list(shape)
    # Check broadcast compatibility
    if len(shape) < self.ndim:
        raise ValueError(f"Cannot expand to shape {shape} from {self.shape}")
    for i in range(1, self.ndim + 1):
        if self.shape[-i] != 1 and self.shape[-i] != shape[-i]:
            raise ValueError(f"Cannot expand shape {self.shape} to {shape}")
    # In C expand kernel, the dims are passed or derived. Let's just create node.
    return _run_op([self], OpType.EXPAND, shape)

def backward_op(self):
    if len(self.shape) == 1 and self.shape[0] == 1:
        set_ones_grad(self)
    backward(self)

def to(self, device):
    if self.device == device:
        return self
    from ._internal import get_arenas, get_persistent_arenas, Parameter
    is_param = isinstance(self, Parameter)
    meta, data = get_persistent_arenas() if is_param else get_arenas()
    t = tensor_init(meta, data, device, self.dtype, list(self.shape), self.requires_grad)
    t.copy_from_numpy(self.numpy())
    if is_param:
        t.__class__ = Parameter
    return t

def item(self):
    if np.prod(self.shape) != 1:
        raise ValueError("only single-element tensors can be converted to Python scalars")
    return float(self.numpy().flatten()[0])

def size(self):
    return self.shape

def __repr__(self):
    np_str = str(self.numpy())
    device_str = ", device=CUDA" if self.device == Device.CUDA else ""
    grad_str = ", requires_grad=True" if self.requires_grad else ""
    return f"tensor({np_str}{device_str}{grad_str})"

# Monkeypatch
Tensor.__add__ = add
Tensor.__radd__ = add
Tensor.__sub__ = sub
Tensor.__rsub__ = rsub
Tensor.__mul__ = mul
Tensor.__rmul__ = mul
Tensor.__truediv__ = div
Tensor.__rtruediv__ = rdiv
Tensor.__neg__ = neg
Tensor.__abs__ = abs_op
Tensor.__matmul__ = matmul

Tensor.log = log
Tensor.exp = exp
Tensor.sin = sin
Tensor.cos = cos
Tensor.tan = tan
Tensor.sum = sum
Tensor.mean = mean
Tensor.min = min
Tensor.max = max
Tensor.view = view
Tensor.reshape = reshape
Tensor.transpose = transpose
Tensor.squeeze = squeeze
Tensor.unsqueeze = unsqueeze
Tensor.flatten = flatten
Tensor.expand = expand
Tensor.backward = backward_op
Tensor.to = to
Tensor.item = item
Tensor.size = size
Tensor.__repr__ = __repr__
Tensor.__str__ = __repr__
