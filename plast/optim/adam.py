from .optimizer import Optimizer
from ..plast_core import Adam as Adam_core, adam_step_cpu, Device, Arena
from .._internal import get_persistent_arenas

class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps})
        
        # We need an optimizer arena on CPU to store optimizer states
        # 10MB should be plenty for storing momentum pointers and states
        from ..plast_core import CPU
        self._opt_arena = Arena(10 * 1024 * 1024, CPU)
        
        # Get data arena
        _, persistent_data_arena = get_persistent_arenas()
            
        beta1, beta2 = betas
        self._adam = Adam_core(self._opt_arena, persistent_data_arena, lr, beta1, beta2, eps)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            
            # Sync hyperparameters if changed
            self._adam.lr = lr
            self._adam.beta1 = beta1
            self._adam.beta2 = beta2
            self._adam.epsilon = eps
            
            # We filter CPU and CUDA params
            cuda_params = [p for p in group["params"] if p.device == Device.CUDA]
            cpu_params = [p for p in group["params"] if p.device == Device.CPU]
            
            if cuda_params:
                raise NotImplementedError("Adam CUDA is not implemented in the C engine, please use CPU device")
                
            if cpu_params:
                adam_step_cpu(self._adam, cpu_params)

    def __del__(self):
        # Release the optimizer arena
        if hasattr(self, "_opt_arena"):
            self._opt_arena.release()
