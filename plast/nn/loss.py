from .module import Module
from . import functional as F


class MSELoss(Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        return F.mse_loss(input, target, reduction=self.reduction)

    def __repr__(self):
        return f"MSELoss(reduction='{self.reduction}')"


class L1Loss(Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        return F.l1_loss(input, target, reduction=self.reduction)

    def __repr__(self):
        return f"L1Loss(reduction='{self.reduction}')"


class CrossEntropyLoss(Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        return F.cross_entropy(input, target, reduction=self.reduction)

    def __repr__(self):
        return f"CrossEntropyLoss(reduction='{self.reduction}')"


class BCELoss(Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        # -mean(target * log(input) + (1 - target) * log(1 - input))
        log_input = input.log()
        one_minus_input = 1.0 - input
        log_one_minus_input = one_minus_input.log()

        loss = -(target * log_input + (1.0 - target) * log_one_minus_input)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def __repr__(self):
        return f"BCELoss(reduction='{self.reduction}')"


class BCEWithLogitsLoss(Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        # BCEWithLogitsLoss(x, y) = BCELoss(sigmoid(x), y)
        sig = F.sigmoid(input)

        log_input = sig.log()
        one_minus_input = 1.0 - sig
        log_one_minus_input = one_minus_input.log()

        loss = -(target * log_input + (1.0 - target) * log_one_minus_input)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def __repr__(self):
        return f"BCEWithLogitsLoss(reduction='{self.reduction}')"
