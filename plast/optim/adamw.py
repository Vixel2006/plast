from .optimizer import Optimizer
from ..plast_core import AdamW as AdamW_core, adamw_step_cpu, Device, Arena
from .._internal import get_persistent_arenas


class AdamW(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(
            params, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        )

        from ..plast_core import CPU

        self._opt_arena = Arena(10 * 1024 * 1024, CPU)

        _, persistent_data_arena = get_persistent_arenas()

        beta1, beta2 = betas
        self._adamw = AdamW_core(
            self._opt_arena, persistent_data_arena, lr, beta1, beta2, eps, weight_decay
        )

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            # Sync hyperparameters if changed
            self._adamw.lr = lr
            self._adamw.beta1 = beta1
            self._adamw.beta2 = beta2
            self._adamw.epsilon = eps
            self._adamw.weight_decay = wd

            cuda_params = [p for p in group["params"] if p.device == Device.CUDA]
            cpu_params = [p for p in group["params"] if p.device == Device.CPU]

            if cuda_params:
                raise NotImplementedError(
                    "AdamW CUDA is not implemented in the C engine, please use CPU device"
                )

            if cpu_params:
                adamw_step_cpu(self._adamw, cpu_params)

    def __del__(self):
        if hasattr(self, "_opt_arena"):
            self._opt_arena.release()
