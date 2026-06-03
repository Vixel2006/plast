import numpy as np

class _LRScheduler:
    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param_groups doesn't have initial_lr key")
        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]
        self.last_epoch = last_epoch
        self.step()

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
            values = self.get_lr()
        else:
            self.last_epoch = epoch
            values = self.get_lr()

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            group, lr = data
            group['lr'] = lr

class StepLR(_LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]

class MultiStepLR(_LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        self.milestones = set(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]

class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]

class CosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - np.cos(np.pi / self.T_max)) / 2
                    for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)]
        return [self.eta_min + (group['lr'] - self.eta_min) *
                (1 + np.cos(np.pi * self.last_epoch / self.T_max)) /
                (1 + np.cos(np.pi * (self.last_epoch - 1) / self.T_max))
                for group in self.optimizer.param_groups]

class ReduceLROnPlateau:
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.min_lrs = [min_lr] * len(optimizer.param_groups)
        self.eps = eps
        self.mode = mode
        self.patience_counter = 0
        self.best = None
        self.num_bad_epochs = 0
        self.last_epoch = 0
        self._init_is_better(mode, threshold, threshold_mode)
        self._reset()

    def _reset(self):
        self.best = np.inf if self.mode == 'min' else -np.inf
        self.num_bad_epochs = 0
        self.cooldown_counter = 0

    def step(self, metrics, epoch=None):
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _reduce_lr(self, epoch):
        for i, group in enumerate(self.optimizer.param_groups):
            old_lr = float(group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                group['lr'] = new_lr
                print(f"Epoch {epoch}: reducing learning rate of group {i} to {new_lr:.4e}.")

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode == 'min':
            self.is_better = lambda a, best: a < best - threshold
        else:
            self.is_better = lambda a, best: a > best + threshold
