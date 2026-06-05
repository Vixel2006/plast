from .optimizer import Optimizer
from ..plast_core import SGD as SGD_core, sgd_step_cpu, Device

try:
    from ..plast_core import sgd_step_cuda
except ImportError:
    sgd_step_cuda = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01):
        super().__init__(params, {"lr": lr})
        self._sgd = SGD_core(lr)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            self._sgd.lr = lr

            cuda_params = [p._t for p in group["params"] if p.device == Device.CUDA]
            cpu_params = [p._t for p in group["params"] if p.device == Device.CPU]

            if cuda_params:
                if sgd_step_cuda is not None:
                    sgd_step_cuda(self._sgd, cuda_params)
            if cpu_params:
                sgd_step_cpu(self._sgd, cpu_params)
