from .optimizer import Optimizer
from ..plast_core import SGD as SGD_core, sgd_step_cpu, sgd_step_cuda, Device

class SGD(Optimizer):
    def __init__(self, params, lr=0.01):
        super().__init__(params, {"lr": lr})
        self._sgd = SGD_core(lr)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            # If lr has changed from the initial default, we update SGD_core lr
            # Since C SGD struct only has lr, we can just update it
            self._sgd.lr = lr
            
            cuda_params = [p for p in group["params"] if p.device == Device.CUDA]
            cpu_params = [p for p in group["params"] if p.device == Device.CPU]
            
            if cuda_params:
                sgd_step_cuda(self._sgd, cuda_params)
            if cpu_params:
                sgd_step_cpu(self._sgd, cpu_params)
