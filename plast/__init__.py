from .plast_core import Arena, Tensor, Device, DType, OpType, forward, backward, SGD
import numpy as np

# Global arenas for convenience if needed, but user should manage them
_meta_arena = None
_data_arena = None

def init_arenas(meta_size_mb=10, data_size_mb=100, device=Device.CPU):
    global _meta_arena, _data_arena
    from .plast_core import CPU
    _meta_arena = Arena(meta_size_mb * 1024 * 1024, CPU)
    _data_arena = Arena(data_size_mb * 1024 * 1024, device)
    return _meta_arena, _data_arena

def tensor(data, device=Device.CPU, dtype=DType.Float32, requires_grad=False):
    if _meta_arena is None or _data_arena is None:
        raise RuntimeError("Arenas not initialized. Call plast.init_arenas() first.")
    
    data = np.array(data, dtype=np.float32)
    shape = list(data.shape)
    
    from .plast_core import tensor_init
    t = tensor_init(_meta_arena, _data_arena, device, dtype, shape, requires_grad)
    t.copy_from_numpy(data)
    return t

class Parameter(Tensor):
    # This might need care because Tensor is a pybind11 class
    pass
