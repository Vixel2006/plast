from .module import Module
from . import functional as F


class MSELoss(Module):
    """Mean Squared Error loss: ``mean((input - target)²)``.

    Args:
        reduction: ``'mean'`` (default), ``'sum'``, or ``'none'``.

    Example::

        loss_fn = plast.nn.MSELoss()
        loss = loss_fn(predictions, targets)
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        return F.mse_loss(input, target, reduction=self.reduction)

    def extra_repr(self) -> str:
        return f"reduction='{self.reduction}'"

    def __repr__(self):
        return f"MSELoss({self.extra_repr()})"


class L1Loss(Module):
    """Mean Absolute Error (L1) loss: ``mean(|input - target|)``.

    Args:
        reduction: ``'mean'`` (default), ``'sum'``, or ``'none'``.

    Example::

        loss_fn = plast.nn.L1Loss()
        loss = loss_fn(predictions, targets)
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        return F.l1_loss(input, target, reduction=self.reduction)

    def extra_repr(self) -> str:
        return f"reduction='{self.reduction}'"

    def __repr__(self):
        return f"L1Loss({self.extra_repr()})"


class SmoothL1Loss(Module):
    """Huber / smooth-L1 loss.

    Behaves like L2 for ``|x| < beta`` and like L1 otherwise, providing a
    smooth transition at zero that is less sensitive to outliers than MSE.

    Args:
        beta:      Threshold at which the loss switches from L2 to L1 (default 1.0).
        reduction: ``'mean'`` (default), ``'sum'``, or ``'none'``.

    Example::

        loss_fn = plast.nn.SmoothL1Loss(beta=0.5)
    """

    def __init__(self, beta: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, input, target):
        return F.smooth_l1_loss(input, target, beta=self.beta, reduction=self.reduction)

    def extra_repr(self) -> str:
        return f"beta={self.beta}, reduction='{self.reduction}'"

    def __repr__(self):
        return f"SmoothL1Loss({self.extra_repr()})"


class CrossEntropyLoss(Module):
    """Cross-entropy loss combining log-softmax and NLL.

    Accepts unnormalised logits directly — no need to apply softmax first.

    Args:
        reduction: ``'mean'`` (default), ``'sum'``, or ``'none'``.

    Example::

        loss_fn = plast.nn.CrossEntropyLoss()
        loss = loss_fn(logits, targets)  # targets can be class indices [N] or one-hot [N, C]
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        return F.cross_entropy(input, target, reduction=self.reduction)

    def extra_repr(self) -> str:
        return f"reduction='{self.reduction}'"

    def __repr__(self):
        return f"CrossEntropyLoss({self.extra_repr()})"


class NLLLoss(Module):
    """Negative log-likelihood loss.

    Expects *input* to be **log-probabilities** (e.g. output of
    ``plast.nn.functional.log_softmax``).

    Args:
        reduction: ``'mean'`` (default), ``'sum'``, or ``'none'``.

    Example::

        loss_fn = plast.nn.NLLLoss()
        log_probs = plast.nn.functional.log_softmax(logits, dim=-1)
        loss = loss_fn(log_probs, targets)
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        return F.nll_loss(input, target, reduction=self.reduction)

    def extra_repr(self) -> str:
        return f"reduction='{self.reduction}'"

    def __repr__(self):
        return f"NLLLoss({self.extra_repr()})"


class BCELoss(Module):
    """Binary Cross-Entropy loss.

    Expects *input* to be probabilities (i.e., already passed through sigmoid).
    Use :class:`BCEWithLogitsLoss` if your inputs are raw logits.

    Args:
        reduction: ``'mean'`` (default), ``'sum'``, or ``'none'``.

    Example::

        loss_fn = plast.nn.BCELoss()
        probs = plast.nn.functional.sigmoid(logits)
        loss = loss_fn(probs, targets)
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        return F.binary_cross_entropy(input, target, reduction=self.reduction)

    def extra_repr(self) -> str:
        return f"reduction='{self.reduction}'"

    def __repr__(self):
        return f"BCELoss({self.extra_repr()})"


class BCEWithLogitsLoss(Module):
    """BCE loss applied directly to raw logits (numerically more stable).

    Combines sigmoid and BCE in a single numerically stable operation.

    Args:
        reduction: ``'mean'`` (default), ``'sum'``, or ``'none'``.

    Example::

        loss_fn = plast.nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, targets)  # no need to apply sigmoid first
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        return F.binary_cross_entropy_with_logits(input, target, reduction=self.reduction)

    def extra_repr(self) -> str:
        return f"reduction='{self.reduction}'"

    def __repr__(self):
        return f"BCEWithLogitsLoss({self.extra_repr()})"
