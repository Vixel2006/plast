from ..tensor import Tensor
from ..plast_core import Device, zero_grad_cpu

try:
    from ..plast_core import zero_grad_cuda
except ImportError:
    zero_grad_cuda = None


class Module:
    def __init__(self):
        self.training = True
        self._modules = {}
        self._parameters = {}

    def __setattr__(self, name, value):
        if isinstance(value, Tensor):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self):
        params = list(self._parameters.values())
        for m in self._modules.values():
            params.extend(m.parameters())
        return params

    def named_parameters(self, prefix=""):
        named_params = []
        for name, param in self._parameters.items():
            named_params.append((f"{prefix}{name}", param))
        for m_name, m in self._modules.items():
            named_params.extend(m.named_parameters(f"{prefix}{m_name}."))
        return named_params

    def train(self, mode=True):
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def zero_grad(self):
        for p in self.parameters():
            raw = p._t
            if p.device == Device.CUDA:
                if zero_grad_cuda is not None:
                    zero_grad_cuda(raw)
            else:
                zero_grad_cpu(raw)

    def state_dict(self, prefix=""):
        state = {}
        for name, param in self._parameters.items():
            state[f"{prefix}{name}"] = param.numpy()
        for m_name, m in self._modules.items():
            state.update(m.state_dict(f"{prefix}{m_name}."))
        return state

    def load_state_dict(self, state_dict, prefix=""):
        for name, param in self._parameters.items():
            key = f"{prefix}{name}"
            if key in state_dict:
                param.copy_from_numpy(state_dict[key])
        for m_name, m in self._modules.items():
            m.load_state_dict(state_dict, f"{prefix}{m_name}.")
