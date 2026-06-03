import numpy as np
from .module import Module
from .._internal import Parameter, Device
from . import functional as F


class Conv2d(Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=Device.CPU
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)

        self.stride = stride

        # Initialize kernel weight parameter: [out_channels, in_channels, kh, kw]
        kh, kw = self.kernel_size
        fan_in = in_channels * kh * kw
        w_bound = np.sqrt(2.0 / fan_in)
        w_data = np.random.randn(out_channels, in_channels, kh, kw).astype(np.float32) * w_bound
        self.weight = Parameter(w_data, device=device)

        if bias:
            # Bias shape: [out_channels]
            b_data = np.zeros((out_channels,), dtype=np.float32)
            self.bias = Parameter(b_data, device=device)
        else:
            self.bias = None

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, stride=self.stride)

    def __repr__(self):
        return f"Conv2d({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, bias={self.bias is not None})"
