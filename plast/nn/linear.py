import numpy as np
from .module import Module
from .._internal import Parameter, Device
from . import functional as F


class Linear(Module):
    """Applies a linear transformation: ``y = x @ W + b``.

    Args:
        in_features:  Size of each input sample.
        out_features: Size of each output sample.
        bias:         If ``True`` (default), adds a learnable bias term.
        device:       Device to allocate parameters on.

    Shape:
        - Input:  ``[*, in_features]``
        - Output: ``[*, out_features]``

    Example::

        fc = plast.nn.Linear(784, 256)
        out = fc(x)  # [batch, 256]
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=Device.CPU):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Kaiming / He initialization for layers followed by ReLU
        w_bound = np.sqrt(2.0 / in_features)
        w_data = np.random.randn(in_features, out_features).astype(np.float32) * w_bound
        self.weight = Parameter(w_data, device=device)

        if bias:
            b_data = np.zeros((1, out_features), dtype=np.float32)
            self.bias = Parameter(b_data, device=device)
        else:
            self.bias = None

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )

    def __repr__(self) -> str:
        return f"Linear({self.extra_repr()})"
