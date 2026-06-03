import numpy as np
from .plast_core import Arena, Tensor, Device, DType, tensor_init

_meta_arena = None
_data_arena = None
_persistent_meta_arena = None
_persistent_data_arena = None

def init_arenas(meta_size_mb=10, data_size_mb=100, device=Device.CPU):
    global _meta_arena, _data_arena, _persistent_meta_arena, _persistent_data_arena
    from .plast_core import CPU
    
    _persistent_meta_arena = Arena(meta_size_mb * 1024 * 1024, CPU)
    _persistent_data_arena = Arena(data_size_mb * 1024 * 1024, device)
    
    _meta_arena = Arena(meta_size_mb * 1024 * 1024, CPU)
    _data_arena = Arena(data_size_mb * 1024 * 1024, device)
    
    return _meta_arena, _data_arena

def get_arenas():
    if _meta_arena is None or _data_arena is None:
        raise RuntimeError("Arenas not initialized. Call plast.init_arenas() first.")
    return _meta_arena, _data_arena

def get_persistent_arenas():
    if _persistent_meta_arena is None or _persistent_data_arena is None:
        raise RuntimeError("Arenas not initialized. Call plast.init_arenas() first.")
    return _persistent_meta_arena, _persistent_data_arena

def reset_transient_arenas():
    global _meta_arena, _data_arena
    if _meta_arena is not None:
        _meta_arena.reset()
    if _data_arena is not None:
        _data_arena.reset()

def tensor(data, device=Device.CPU, dtype=DType.Float32, requires_grad=False, persistent=False):
    if persistent:
        meta, data_arena = get_persistent_arenas()
    else:
        meta, data_arena = get_arenas()
        
    data = np.array(data, dtype=np.float32)
    shape = list(data.shape)
    
    t = tensor_init(meta, data_arena, device, dtype, shape, requires_grad)
    t.copy_from_numpy(data)
    return t

def Parameter(data, device=Device.CPU, dtype=DType.Float32, requires_grad=True):
    return tensor(data, device=device, dtype=dtype, requires_grad=requires_grad, persistent=True)


