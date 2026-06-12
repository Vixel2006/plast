"""plast.nn.functional — stateless functional operations.

All functions here are pure: they take tensors as input and return
tensors without maintaining any internal state.  They are the building
blocks for both the :class:`~plast.nn.Module` layer classes and direct
functional usage::

    import plast.nn.functional as F

    out  = F.relu(x)
    loss = F.cross_entropy(logits, targets)
"""

import math
import numpy as np
from ..tensor import Tensor
from ..plast_core import OpType, Device
from .._internal import tensor
from ..tensor import _run_op


# ── Activations ──

def relu(x: Tensor) -> Tensor:
    """Rectified Linear Unit: ``max(0, x)``."""
    return leaky_relu(x, negative_slope=0.0)


def leaky_relu(x: Tensor, negative_slope: float = 0.01) -> Tensor:
    """Leaky ReLU: ``x if x >= 0 else negative_slope * x``."""
    return _run_op(
        [x],
        OpType.LEAKY_RELU,
        list(x.shape),
        dim=0,
        keepdim=0,
        fval=negative_slope,
        requires_grad=x.requires_grad,
    )


def sigmoid(x: Tensor) -> Tensor:
    """Element-wise sigmoid: ``1 / (1 + exp(-x))``."""
    neg_x = -x
    exp_neg_x = neg_x.exp()
    denom = exp_neg_x + 1.0
    return denom.rdiv(1.0)


def tanh(x: Tensor) -> Tensor:
    """Element-wise hyperbolic tangent."""
    exp_x = x.exp()
    exp_neg_x = (-x).exp()
    num = exp_x - exp_neg_x
    denom = exp_x + exp_neg_x
    return num / denom


def gelu(x: Tensor) -> Tensor:
    """Gaussian Error Linear Unit (tanh approximation).

    ``x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))``
    """
    c = math.sqrt(2.0 / math.pi)
    # x^3 via exp(3*log|x|) is unstable for negative x; use x*x*x via numpy path
    # We compose with existing ops: tanh( c*(x + 0.044715 * x*x*x) )
    x3 = x * x * x
    inner = (x + x3 * 0.044715) * c
    return x * (tanh(inner) + 1.0) * 0.5


def silu(x: Tensor) -> Tensor:
    """Sigmoid Linear Unit (Swish): ``x * sigmoid(x)``."""
    return x * sigmoid(x)


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Numerically stable softmax along *dim*.

    ``softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))``
    """
    max_x = x.max(dim=dim, keepdim=True)
    shifted_x = x - max_x
    exp_x = shifted_x.exp()
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / sum_exp


def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Numerically stable log-softmax along *dim*.

    ``log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))``
    """
    max_x = x.max(dim=dim, keepdim=True)
    shifted_x = x - max_x
    log_sum_exp = shifted_x.exp().sum(dim=dim, keepdim=True).log()
    return shifted_x - log_sum_exp


# ── Dropout ──

def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """Randomly zero elements with probability *p* during training.

    The remaining elements are scaled by ``1 / (1 - p)``.
    """
    if not training or p <= 0.0:
        return x
    if p >= 1.0:
        raise ValueError(
            f"dropout probability must be in [0, 1), got {p}."
        )
    mask_np = (np.random.rand(*x.shape) >= p).astype(np.float32) / (1.0 - p)
    mask = tensor(mask_np, device=x.device)
    return x * mask


# ── Linear ──

def linear(input: Tensor, weight: Tensor, bias=None) -> Tensor:
    """Apply a linear transformation: ``input @ weight + bias``.

    Args:
        input:  ``[batch, in_features]``
        weight: ``[in_features, out_features]``
        bias:   ``[1, out_features]`` or ``None``
    """
    out = input @ weight
    if bias is not None:
        out = out + bias
    return out


# ── Convolution ──

def conv2d(input: Tensor, weight: Tensor, bias=None, stride: int = 1) -> Tensor:
    """Apply a 2-D convolution.

    Args:
        input:  ``[N, C_in, H, W]``
        weight: ``[C_out, C_in, KH, KW]``
        bias:   ``[C_out]`` or ``None``
        stride: Convolution stride (default 1).
    """
    N, C, H, W = input.shape
    F_out, _, KH, KW = weight.shape
    H_out = (H - KH) // stride + 1
    W_out = (W - KW) // stride + 1

    out_shape = [N * H_out * W_out, F_out]
    out = _run_op([input, weight], OpType.CONV2D, out_shape, dim=0, keepdim=stride)

    # Reshape from [N*H_out*W_out, F_out] → [N, F_out, H_out, W_out]
    out_reshaped = out.view(N, H_out, W_out, F_out)
    out_transposed = out_reshaped.transpose(1, 3).transpose(2, 3)

    if bias is not None:
        bias_broadcast = bias.view(1, F_out, 1, 1)
        out_transposed = out_transposed + bias_broadcast

    return out_transposed


# ── Normalization ──

def batch_norm(
    input: Tensor,
    running_mean,
    running_var,
    weight=None,
    bias=None,
    training: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    """Batch normalisation for 2-D ``[N, C]`` or 4-D ``[N, C, H, W]`` inputs."""
    ndim = input.ndim
    if ndim == 2:
        # [N, C]
        if training:
            mean = input.mean(dim=0, keepdim=True)
            diff = input - mean
            var = (diff * diff).mean(dim=0, keepdim=True)

            mean_np = mean.numpy().flatten()
            var_np = var.numpy().flatten()
            running_mean.copy_from_numpy(
                (1.0 - momentum) * running_mean.numpy() + momentum * mean_np
            )
            running_var.copy_from_numpy(
                (1.0 - momentum) * running_var.numpy() + momentum * var_np
            )
        else:
            mean = tensor(running_mean.numpy().reshape(1, input.shape[1]), device=input.device)
            var = tensor(running_var.numpy().reshape(1, input.shape[1]), device=input.device)

        std = ((var + eps).log() * 0.5).exp()
        norm_input = (input - mean) / std

        if weight is not None:
            norm_input = norm_input * weight.view(1, input.shape[1])
        if bias is not None:
            norm_input = norm_input + bias.view(1, input.shape[1])
        return norm_input

    elif ndim == 4:
        # [N, C, H, W]
        N, C, H, W = input.shape
        permuted = input.transpose(0, 1)  # [C, N, H, W]
        flat = permuted.reshape(C, N * H * W)

        if training:
            mean = flat.mean(dim=1, keepdim=True)
            diff = flat - mean
            var = (diff * diff).mean(dim=1, keepdim=True)

            mean_np = mean.numpy().flatten()
            var_np = var.numpy().flatten()
            running_mean.copy_from_numpy(
                (1.0 - momentum) * running_mean.numpy() + momentum * mean_np
            )
            running_var.copy_from_numpy(
                (1.0 - momentum) * running_var.numpy() + momentum * var_np
            )
        else:
            mean = tensor(running_mean.numpy().reshape(C, 1), device=input.device)
            var = tensor(running_var.numpy().reshape(C, 1), device=input.device)

        std = ((var + eps).log() * 0.5).exp()
        norm_flat = (flat - mean) / std

        norm_permuted = norm_flat.reshape(C, N, H, W)
        norm_input = norm_permuted.transpose(0, 1)  # back to [N, C, H, W]

        if weight is not None:
            norm_input = norm_input * weight.view(1, C, 1, 1)
        if bias is not None:
            norm_input = norm_input + bias.view(1, C, 1, 1)
        return norm_input
    else:
        raise ValueError(
            f"batch_norm only supports 2-D [N, C] or 4-D [N, C, H, W] inputs, "
            f"got {ndim}-D input."
        )


def layer_norm(
    input: Tensor,
    normalized_shape,
    weight=None,
    bias=None,
    eps: float = 1e-5,
) -> Tensor:
    """Layer normalisation over the last ``len(normalized_shape)`` dimensions."""
    input_shape = list(input.shape)
    norm_ndim = len(normalized_shape)

    num_features = int(np.prod(normalized_shape))
    batch_size = int(np.prod(input_shape[:-norm_ndim])) if norm_ndim < len(input_shape) else 1

    flat = input.reshape(batch_size, num_features)
    mean = flat.mean(dim=1, keepdim=True)
    diff = flat - mean
    var = (diff * diff).mean(dim=1, keepdim=True)
    std = (var + eps).log().mul(0.5).exp()

    norm_flat = (flat - mean) / std
    norm_input = norm_flat.reshape(input_shape)

    if weight is not None:
        norm_input = norm_input * weight
    if bias is not None:
        norm_input = norm_input + bias
    return norm_input


# ── Loss functions ──

def mse_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """Mean Squared Error loss: ``mean((input - target)²)``."""
    _check_reduction(reduction)
    diff = input - target
    sq_diff = diff * diff
    return _reduce(sq_diff, reduction)


def l1_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """L1 (mean absolute error) loss: ``mean(|input - target|)``."""
    _check_reduction(reduction)
    diff = abs(input - target)
    return _reduce(diff, reduction)


def smooth_l1_loss(input: Tensor, target: Tensor, beta: float = 1.0, reduction: str = "mean") -> Tensor:
    """Huber / smooth-L1 loss.

    Behaves like L2 for ``|x| < beta`` and L1 otherwise, smoothing the
    transition at zero.
    """
    _check_reduction(reduction)
    diff_np = (input - target).numpy()
    abs_diff = np.abs(diff_np)
    loss_np = np.where(
        abs_diff < beta,
        0.5 * diff_np ** 2 / beta,
        abs_diff - 0.5 * beta,
    ).astype(np.float32)
    loss = tensor(loss_np, device=input.device)
    return _reduce(loss, reduction)


def cross_entropy(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """Cross-entropy loss with numerically stable log-softmax.

    Args:
        input:     Unnormalised logits ``[N, C]``.
        target:    Class indices (1-D, shape ``[N]``) **or** one-hot
                   probabilities ``[N, C]``.
        reduction: ``'mean'`` (default), ``'sum'``, or ``'none'``.
    """
    _check_reduction(reduction)
    log_soft = log_softmax(input, dim=1)

    # Support integer class indices as 1-D target
    if target.ndim == 1:
        num_classes = input.shape[1]
        t_np = target.numpy().astype(np.int32)
        one_hot_np = np.zeros((t_np.shape[0], num_classes), dtype=np.float32)
        one_hot_np[np.arange(t_np.shape[0]), t_np] = 1.0
        target = tensor(one_hot_np, device=input.device)

    loss = -(target * log_soft).sum(dim=1)
    return _reduce(loss, reduction)


def nll_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """Negative log-likelihood loss.

    Expects *input* to be **log-probabilities** (e.g. output of
    :func:`log_softmax`).

    Args:
        input:     Log-probabilities ``[N, C]``.
        target:    Class indices ``[N]`` (integers stored as float32).
        reduction: ``'mean'``, ``'sum'``, or ``'none'``.
    """
    _check_reduction(reduction)
    t_np = target.numpy().astype(np.int32)
    N = t_np.shape[0]
    # Gather the log-prob for the correct class
    lp_np = input.numpy()
    selected = lp_np[np.arange(N), t_np].astype(np.float32)
    loss = tensor(-selected, device=input.device)
    return _reduce(loss, reduction)


def binary_cross_entropy(
    input: Tensor, target: Tensor, reduction: str = "mean"
) -> Tensor:
    """BCE loss: ``-mean(t·log(p) + (1-t)·log(1-p))``.

    Expects *input* to be probabilities (already passed through sigmoid).
    Use :func:`binary_cross_entropy_with_logits` for raw logits.
    """
    _check_reduction(reduction)
    log_input = input.log()
    log_one_minus = (1.0 - input).log()
    loss = -(target * log_input + (1.0 - target) * log_one_minus)
    return _reduce(loss, reduction)


def binary_cross_entropy_with_logits(
    input: Tensor, target: Tensor, reduction: str = "mean"
) -> Tensor:
    """BCE loss applied to raw logits (numerically more stable)."""
    _check_reduction(reduction)
    sig = sigmoid(input)
    return binary_cross_entropy(sig, target, reduction=reduction)


# ── Private helpers ──

def _check_reduction(reduction: str) -> None:
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(
            f"Invalid reduction mode '{reduction}'. "
            "Expected one of: 'mean', 'sum', 'none'."
        )


def _reduce(loss: Tensor, reduction: str) -> Tensor:
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss
