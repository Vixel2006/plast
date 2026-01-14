from .plast_core import OpType, create_node, tensor_init, numel, zeros, Device, DType
from . import _meta_arena, _data_arena
import numpy as np

def _get_arenas():
    if _meta_arena is None or _data_arena is None:
        raise RuntimeError("Arenas not initialized. Call plast.init_arenas() first.")
    return _meta_arena, _data_arena

class Module:
    def __init__(self):
        self.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self):
        params = []
        for name, value in self.__dict__.items():
            if isinstance(value, Module):
                params.extend(value.parameters())
            elif name == "weight" or name == "bias":
                params.append(value)
        return params

class Linear(Module):
    def __init__(self, in_features, out_features, device=Device.CPU):
        super().__init__()
        meta, data = _get_arenas()
        
        self.weight = tensor_init(meta, data, device, DType.Float32, [in_features, out_features], True)
        self.bias = tensor_init(meta, data, device, DType.Float32, [1, out_features], True)
        
        # Xavier-like init
        w_data = np.random.randn(in_features, out_features).astype(np.float32) * np.sqrt(2.0 / in_features)
        self.weight.copy_from_numpy(w_data)
        zeros(self.bias)

    def forward(self, x):
        meta, data = _get_arenas()
        device = x.device
        
        # Matmul output
        mm_shape = [x.shape[0], self.weight.shape[1]]
        mm_out = tensor_init(meta, data, device, DType.Float32, mm_shape, True)
        create_node(meta, [x, self.weight], mm_out, OpType.MATMUL, 0, False)
        
        # Add bias
        out = tensor_init(meta, data, device, DType.Float32, mm_shape, True)
        create_node(meta, [mm_out, self.bias], out, OpType.ADD, 0, False)
        
        return out

class ReLU(Module):
    def forward(self, x):
        meta, data = _get_arenas()
        device = x.device
        shape = x.shape
        
        # (x + abs(x)) / 2
        # abs
        x_abs = tensor_init(meta, data, device, DType.Float32, shape, True)
        create_node(meta, [x], x_abs, OpType.ABS, 0, False)
        
        # add
        x_plus_abs = tensor_init(meta, data, device, DType.Float32, shape, True)
        create_node(meta, [x, x_abs], x_plus_abs, OpType.ADD, 0, False)
        
        # div by 2
        two = tensor_init(meta, data, device, DType.Float32, [1], False)
        two.copy_from_numpy(np.array([2.0], dtype=np.float32))
        
        out = tensor_init(meta, data, device, DType.Float32, shape, True)
        create_node(meta, [x_plus_abs, two], out, OpType.DIV, 0, False)
        
        return out

class MSELoss(Module):
    def forward(self, input, target):
        meta, data = _get_arenas()
        device = input.device
        shape = input.shape
        
        # (input - target)^2
        # sub
        diff = tensor_init(meta, data, device, DType.Float32, shape, True)
        create_node(meta, [input, target], diff, OpType.SUB, 0, False)
        
        # mul (sq)
        sq_diff = tensor_init(meta, data, device, DType.Float32, shape, True)
        create_node(meta, [diff, diff], sq_diff, OpType.MUL, 0, False)
        
        # mean
        loss = tensor_init(meta, data, device, DType.Float32, [1], True)
        # MAX_NDIM + 1 = 9 in our C code
        create_node(meta, [sq_diff], loss, OpType.MEAN, 9, False)
        
        return loss
