import numpy as np
from ..plast_core import LRScheduler as _CLRScheduler, LRSchedulerType
from ..plast_core import ReduceLROnPlateau as _CReduceLROnPlateau, ReduceLROnPlateauMode


class _LRScheduler:
    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if "initial_lr" not in group:
                    raise KeyError("param_groups doesn't have initial_lr key")
        self.base_lrs = [group["initial_lr"] for group in optimizer.param_groups]
        self.last_epoch = last_epoch
        self._init_c_backend()
        self.step()

    def state_dict(self):
        state = {key: value for key, value in self.__dict__.items() if key not in ("optimizer", "_scheduler")}
        state["last_epoch"] = self._scheduler.last_epoch
        return state

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._scheduler.last_epoch = self.last_epoch

    def _init_c_backend(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
            epoch_val = -1
        else:
            self.last_epoch = epoch
            epoch_val = epoch

        current_lrs = [group["lr"] for group in self.optimizer.param_groups]
        new_lrs = self._scheduler.step(current_lrs, epoch_val)

        for group, lr in zip(self.optimizer.param_groups, new_lrs):
            group["lr"] = lr


class StepLR(_LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def _init_c_backend(self):
        self._scheduler = _CLRScheduler(
            LRSchedulerType.STEP_LR,
            self.base_lrs,
            self.step_size,
            self.gamma,
            self.last_epoch
        )


class MultiStepLR(_LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        self.milestones = milestones
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def _init_c_backend(self):
        milestones_list = sorted(list(self.milestones))
        self._scheduler = _CLRScheduler(
            self.base_lrs,
            milestones_list,
            self.gamma,
            self.last_epoch
        )


class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def _init_c_backend(self):
        self._scheduler = _CLRScheduler(
            LRSchedulerType.EXPONENTIAL_LR,
            self.base_lrs,
            0,
            self.gamma,
            self.last_epoch
        )


class CosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def _init_c_backend(self):
        self._scheduler = _CLRScheduler(
            LRSchedulerType.COSINE_ANNEALING_LR,
            self.base_lrs,
            self.T_max,
            self.eta_min,
            self.last_epoch
        )


class ReduceLROnPlateau:
    def __init__(
        self,
        optimizer,
        mode="min",
        factor=0.1,
        patience=10,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=1e-8,
    ):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lrs = [min_lr] * len(optimizer.param_groups)
        self.eps = eps
        self.mode = mode
        self.last_epoch = 0

        c_mode = ReduceLROnPlateauMode.MIN_MODE if mode == "min" else ReduceLROnPlateauMode.MAX_MODE
        self._scheduler = _CReduceLROnPlateau(
            len(optimizer.param_groups),
            factor,
            patience,
            threshold,
            cooldown,
            self.min_lrs,
            eps,
            c_mode
        )

    def step(self, metrics, epoch=None):
        if epoch is None:
            epoch_val = -1
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
            epoch_val = epoch

        current_lrs = [group["lr"] for group in self.optimizer.param_groups]
        reduced, new_lrs = self._scheduler.step(float(metrics), current_lrs, epoch_val)

        for i, (group, lr) in enumerate(zip(self.optimizer.param_groups, new_lrs)):
            old_lr = group["lr"]
            group["lr"] = lr
            if reduced and old_lr != lr:
                print(f"Epoch {self.last_epoch}: reducing learning rate of group {i} to {lr:.4e}.")
