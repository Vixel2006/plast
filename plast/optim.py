from .plast_core import SGD as SGD_core, sgd_step_cuda, sgd_step_cpu, zero_grad_cuda, zero_grad_cpu, Device

class Optimizer:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for p in self.params:
            if p.device == Device.CUDA:
                zero_grad_cuda(p)
            else:
                zero_grad_cpu(p)

class SGD(Optimizer):
    def __init__(self, params, lr=0.01):
        super().__init__(params, lr)
        self._sgd = SGD_core(lr)

    def step(self):
        cuda_params = [p for p in self.params if p.device == Device.CUDA]
        cpu_params = [p for p in self.params if p.device == Device.CPU]
        
        if cuda_params:
            sgd_step_cuda(self._sgd, cuda_params)
        if cpu_params:
            sgd_step_cpu(self._sgd, cpu_params)
