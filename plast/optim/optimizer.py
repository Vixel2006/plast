from ..plast_core import Device
from ..plast_core import zero_grad_cpu, zero_grad_cuda

class Optimizer:
    def __init__(self, params, defaults):
        self.params = list(params)
        self.defaults = defaults
        # Initialize param groups
        self.param_groups = [{"params": self.params, **defaults}]

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if p.device == Device.CUDA:
                        zero_grad_cuda(p)
                    else:
                        zero_grad_cpu(p)
