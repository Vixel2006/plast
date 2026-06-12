from .module import Module
from . import functional as F


class ReLU(Module):
    """Applies the Rectified Linear Unit function element-wise: ``max(0, x)``.

    Example::

        act = plast.nn.ReLU()
        out = act(x)  # or equivalently: x.relu()
    """

    def forward(self, input):
        return F.relu(input)

    def __repr__(self):
        return "ReLU()"


class LeakyReLU(Module):
    """Applies LeakyReLU: ``max(α·x, x)`` where α = *negative_slope*.

    Args:
        negative_slope: Slope for negative inputs (default: 0.01).

    Example::

        act = plast.nn.LeakyReLU(0.1)
    """

    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input):
        return F.leaky_relu(input, negative_slope=self.negative_slope)

    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}"

    def __repr__(self):
        return f"LeakyReLU({self.extra_repr()})"


class Sigmoid(Module):
    """Applies the sigmoid function element-wise: ``1 / (1 + exp(-x))``.

    Example::

        act = plast.nn.Sigmoid()
        prob = act(logits)  # or: logits.sigmoid()
    """

    def forward(self, input):
        return F.sigmoid(input)

    def __repr__(self):
        return "Sigmoid()"


class Tanh(Module):
    """Applies the hyperbolic tangent function element-wise.

    Example::

        act = plast.nn.Tanh()
        out = act(x)  # or: x.tanh()
    """

    def forward(self, input):
        return F.tanh(input)

    def __repr__(self):
        return "Tanh()"


class Softmax(Module):
    """Applies softmax over *dim*.

    Args:
        dim: Dimension along which softmax is computed (default: -1).

    Example::

        act = plast.nn.Softmax(dim=-1)
        probs = act(logits)  # or: logits.softmax(dim=-1)
    """

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.softmax(input, dim=self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"

    def __repr__(self):
        return f"Softmax({self.extra_repr()})"


class GELU(Module):
    """Applies the Gaussian Error Linear Unit activation.

    Approximation: ``x * 0.5 * (1 + tanh(√(2/π) · (x + 0.044715·x³)))``.

    Example::

        act = plast.nn.GELU()
        out = act(x)
    """

    def forward(self, input):
        return F.gelu(input)

    def __repr__(self):
        return "GELU()"


class SiLU(Module):
    """Applies the Sigmoid Linear Unit (Swish) activation: ``x * sigmoid(x)``.

    Example::

        act = plast.nn.SiLU()
        out = act(x)
    """

    def forward(self, input):
        return F.silu(input)

    def __repr__(self):
        return "SiLU()"
