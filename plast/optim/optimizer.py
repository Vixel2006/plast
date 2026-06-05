from ..plast_core import Device, zero_grad_cpu

try:
    from ..plast_core import zero_grad_cuda
except ImportError:
    zero_grad_cuda = None


class Optimizer:
    def __init__(self, params, defaults):
        self.params = list(params)
        self.defaults = defaults
        self.param_groups = [{"params": self.params, **defaults}]

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    raw = p._t
                    if p.device == Device.CUDA:
                        if zero_grad_cuda is not None:
                            zero_grad_cuda(raw)
                    else:
                        zero_grad_cpu(raw)
