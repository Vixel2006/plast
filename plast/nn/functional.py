import numpy as np
from ..tensor import Tensor
from ..plast_core import OpType, Device
from .._internal import tensor
from ..tensor import _run_op


def relu(x):
    # C LeakyReLU with alpha = 0.0 is exactly ReLU!
    return leaky_relu(x, negative_slope=0.0)


def leaky_relu(x, negative_slope=0.01):
    return _run_op(
        [x],
        OpType.LEAKY_RELU,
        list(x.shape),
        dim=0,
        keepdim=0,
        fval=negative_slope,
        requires_grad=x.requires_grad,
    )


def sigmoid(x):
    # sigmoid(x) = 1 / (1 + exp(-x))
    neg_x = -x
    exp_neg_x = neg_x.exp()
    denom = exp_neg_x + 1.0
    return denom.rdiv(1.0)


def tanh(x):
    # tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    exp_x = x.exp()
    exp_neg_x = (-x).exp()
    num = exp_x - exp_neg_x
    denom = exp_x + exp_neg_x
    return num / denom


def softmax(x, dim=-1):
    # stable softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    max_x = x.max(dim=dim, keepdim=True)
    shifted_x = x - max_x
    exp_x = shifted_x.exp()
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / sum_exp


def dropout(x, p=0.5, training=True):
    if not training or p <= 0.0:
        return x
    # Generate random mask on CPU, then move to device
    mask_np = (np.random.rand(*x.shape) >= p).astype(np.float32) / (1.0 - p)
    mask = tensor(mask_np, device=x.device)
    return x * mask


def linear(input, weight, bias=None):
    # input: [batch, in_features], weight: [in_features, out_features]
    out = input @ weight
    if bias is not None:
        out = out + bias
    return out


def conv2d(input, weight, bias=None, stride=1):
    # input: [N, C, H, W], weight: [F, C, KH, KW]
    N, C, H, W = input.shape
    F_out, _, KH, KW = weight.shape
    H_out = (H - KH) // stride + 1
    W_out = (W - KW) // stride + 1

    out_shape = [N * H_out * W_out, F_out]
    # We create the node using OpType.CONV2D.
    # The C binding for create_node auto-injects meta-arena address for CONV2D,
    # so we just pass dim=0, keepdim=stride.
    out = _run_op([input, weight], OpType.CONV2D, out_shape, dim=0, keepdim=stride)

    # Reshape from [N * H_out * W_out, F_out] back to [N, F_out, H_out, W_out]
    # To do this, we need transpose:
    # 1. view to [N, H_out, W_out, F_out]
    # 2. transpose axis 3 (F_out) and 1 (H_out) -> [N, F_out, W_out, H_out]
    # 3. transpose axis 2 (W_out) and 3 (H_out) -> [N, F_out, H_out, W_out]
    out_reshaped = out.view(N, H_out, W_out, F_out)
    out_transposed = out_reshaped.transpose(1, 3).transpose(2, 3)

    if bias is not None:
        # bias shape: [1, F_out, 1, 1] for broadcasting
        bias_broadcast = bias.view(1, F_out, 1, 1)
        out_transposed = out_transposed + bias_broadcast

    return out_transposed


def batch_norm(
    input, running_mean, running_var, weight=None, bias=None, training=True, momentum=0.1, eps=1e-5
):
    # Input shape: [N, C, H, W] or [N, C]
    # For simplicity, let's implement standard batch_norm1d / batch_norm2d using mean/var reduction
    ndim = input.ndim
    if ndim == 2:
        # [N, C]
        reduction_dims = [0]
        view_shape = [1, input.shape[1]]
    elif ndim == 4:
        # [N, C, H, W]
        # In Plast, reduction is done per-dimension sequentially.
        # To compute mean across [N, H, W], we can transpose to [C, N, H, W], view to [C, N*H*W],
        # compute mean/var across axis 1, then reshape and transpose back.
        # This is extremely clean and avoids complex C reduction.
        N, C, H, W = input.shape
        # Transpose C to front: [C, N, H, W]
        permuted = input.transpose(
            0, 1
        )  # [C, N, H, W] (wait, transpose takes 2 axes, so transpose(0, 1) swaps 0 and 1)
        flat = permuted.reshape(C, N * H * W)

        if training:
            mean = flat.mean(dim=1, keepdim=True)
            # var = mean((x - mean)^2)
            diff = flat - mean
            var = (diff * diff).mean(dim=1, keepdim=True)

            # Update running statistics (using numpy view since we can update values in running_mean/var)
            mean_np = mean.numpy().flatten()
            var_np = var.numpy().flatten()
            running_mean.copy_from_numpy(
                (1.0 - momentum) * running_mean.numpy() + momentum * mean_np
            )
            running_var.copy_from_numpy((1.0 - momentum) * running_var.numpy() + momentum * var_np)
        else:
            mean = tensor(running_mean.numpy().reshape(C, 1), device=input.device)
            var = tensor(running_var.numpy().reshape(C, 1), device=input.device)

        std = (var + eps).log().mul(0.5).exp()  # sqrt(var + eps)
        norm_flat = (flat - mean) / std

        norm_permuted = norm_flat.reshape(C, N, H, W)
        norm_input = norm_permuted.transpose(0, 1)  # back to [N, C, H, W]

        if weight is not None:
            norm_input = norm_input * weight.view(1, C, 1, 1)
        if bias is not None:
            norm_input = norm_input + bias.view(1, C, 1, 1)
        return norm_input
    else:
        raise ValueError(f"BN only supports 2D or 4D inputs, got {ndim}D")


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    # Normalized shape is the dimensions to normalize over (e.g. last dimensions)
    # LayerNorm: mean and var across the normalized dimensions for each sample
    input_shape = list(input.shape)
    norm_ndim = len(normalized_shape)

    # Reshape input to [batch_size, num_features] where num_features is prod(normalized_shape)
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


def mse_loss(input, target, reduction="mean"):
    diff = input - target
    sq_diff = diff * diff
    if reduction == "mean":
        return sq_diff.mean()
    elif reduction == "sum":
        return sq_diff.sum()
    else:
        return sq_diff


def l1_loss(input, target, reduction="mean"):
    diff = (input - target).abs()
    if reduction == "mean":
        return diff.mean()
    elif reduction == "sum":
        return diff.sum()
    else:
        return diff


def cross_entropy(input, target, reduction="mean"):
    # stable cross entropy: -sum(target * log_softmax(input))
    # log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
    max_x = input.max(dim=1, keepdim=True)
    shifted_x = input - max_x
    log_sum_exp = shifted_x.exp().sum(dim=1, keepdim=True).log()
    log_soft = shifted_x - log_sum_exp

    loss = -(target * log_soft).sum(dim=1)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss
