import numpy as np
from .module import Module
from .._internal import Parameter, Device
from . import functional as F


class BatchNorm1d(Module):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        device=Device.CPU,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        # We need running_mean and running_var as persistent, non-gradient tensors
        from .._internal import tensor

        self.running_mean = tensor(
            np.zeros(num_features, dtype=np.float32), device=device, persistent=True
        )
        self.running_var = tensor(
            np.ones(num_features, dtype=np.float32), device=device, persistent=True
        )

        if affine:
            self.weight = Parameter(np.ones(num_features, dtype=np.float32), device=device)
            self.bias = Parameter(np.zeros(num_features, dtype=np.float32), device=device)
        else:
            self.weight = None
            self.bias = None

    def forward(self, input):
        # BatchNorm1d supports [N, C] or [N, C, L]
        # In functional.py, batch_norm expects 2D [N, C] or 4D [N, C, H, W]
        # So if input is 3D [N, C, L], we can unsqueeze to 4D [N, C, L, 1]
        ndim = input.ndim
        if ndim == 3:
            input_4d = input.unsqueeze(3)
            out_4d = F.batch_norm(
                input_4d,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=self.training,
                momentum=self.momentum,
                eps=self.eps,
            )
            return out_4d.squeeze(3)
        else:
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=self.training,
                momentum=self.momentum,
                eps=self.eps,
            )

    def __repr__(self):
        return f"BatchNorm1d({self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine})"


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, device=Device.CPU):
        super().__init__()
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = Parameter(np.ones(self.normalized_shape, dtype=np.float32), device=device)
            self.bias = Parameter(np.zeros(self.normalized_shape, dtype=np.float32), device=device)
        else:
            self.weight = None
            self.bias = None

    def forward(self, input):
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, eps=self.eps)

    def __repr__(self):
        return f"LayerNorm({self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine})"
