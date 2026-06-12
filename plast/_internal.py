import numpy as np
from contextlib import contextmanager
from .plast_core import Arena, Device, DType, tensor_init


_arena_stack = []
_persistent_meta_arena = None
_persistent_data_arena = None
_default_initialized = False


def _create_arenas(meta_size_mb=10, data_size_mb=100, device=Device.CPU):
    from .plast_core import CPU
    meta = Arena(meta_size_mb * 1024 * 1024, CPU)
    data = Arena(data_size_mb * 1024 * 1024, device)
    return meta, data


def init_arenas(meta_size_mb=64, data_size_mb=512, device=Device.CPU):
    global _persistent_meta_arena, _persistent_data_arena, _default_initialized
    from .plast_core import CPU

    _persistent_meta_arena = Arena(meta_size_mb * 1024 * 1024, CPU)
    _persistent_data_arena = Arena(data_size_mb * 1024 * 1024, device)

    meta, data = _create_arenas(meta_size_mb, data_size_mb, device)
    _arena_stack.append((meta, data))
    _default_initialized = True

    return meta, data


def get_arenas(device=Device.CPU):
    if not _arena_stack:
        init_arenas(device=device)
    return _arena_stack[-1]


def get_persistent_arenas(device=Device.CPU):
    global _persistent_meta_arena, _persistent_data_arena
    if _persistent_meta_arena is None or _persistent_data_arena is None:
        init_arenas(device=device)
    return _persistent_meta_arena, _persistent_data_arena


@contextmanager
def arena_scope(meta_size_mb=10, data_size_mb=100, device=Device.CPU):
    meta, data = _create_arenas(meta_size_mb, data_size_mb, device)
    _arena_stack.append((meta, data))
    try:
        yield meta, data
    finally:
        from .tensor import _get_scheduler
        _get_scheduler().clear_jit()
        meta.reset()
        data.reset()
        meta.release()
        data.release()
        _arena_stack.pop()


def reset_transient_arenas():
    if _arena_stack:
        meta, data = _arena_stack[-1]
        meta.reset()
        data.reset()
    from .tensor import _get_scheduler
    sched = _get_scheduler()
    sched.clear_jit()


def tensor(data, device=Device.CPU, dtype=DType.Float32, requires_grad=False, persistent=False):
    if persistent:
        meta, data_arena = get_persistent_arenas(device)
    else:
        meta, data_arena = get_arenas(device)

    data = np.array(data, dtype=np.float32)
    shape = list(data.shape)

    raw = tensor_init(meta, data_arena, device, dtype, shape, requires_grad)
    raw.copy_from_numpy(data)

    from .tensor import Tensor, Parameter

    cls = Parameter if persistent else Tensor
    t = cls.__new__(cls)
    t._t = raw
    return t


def Parameter(data, device=Device.CPU, dtype=DType.Float32, requires_grad=True):
    return tensor(data, device=device, dtype=dtype, requires_grad=requires_grad, persistent=True)
