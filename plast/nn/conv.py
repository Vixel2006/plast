from os import wait
from .module import Module
from plast.core.tensor import Tensor
from .init import *

class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...] | list[int],
        stride: tuple[int, int] | int = (1, 1),
        padding: int = 0,
        bias: bool = False
    ):
        self.kernel_size = (in_channels, out_channels, *kernel_size)
        self.weights = xavier_normal_((out_channels, in_channels, *kernel_size), in_features=in_channels, out_features=out_channels)
        self.bias = None


        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding

        if bias:
            pass # Placeholder for now

    def forward(self, x: Tensor) -> Tensor:
        out = x # Placeholder for now

        if self.bias:
            out += self.bias
        
        return out

    def reset_parameters(self):
        self.weights = xavier_normal_(self.weights.shape, self.kernel_size[1], self.kernel_size[0])
        if self.bias:
            pass # Placeholder for now

