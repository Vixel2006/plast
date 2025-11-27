from typing import Callable, Union

from plast.core import Tensor
from plast.core.device import Device

from .linear import Linear
from .module import Module


class Pipeline(Module):
    def __init__(self, *layers):
        self.layers = []
        for layer in layers:
            if isinstance(layer, Pipeline):
                self.layers.extend(layer.layers)
            else:
                self.layers.append(layer)

    def __getitem__(
        self, idx: Union[int, slice]
    ) -> Union[Module, Callable, "Pipeline"]:
        if isinstance(idx, slice):
            return Pipeline(*self.layers[idx])
        else:
            return self.layers[idx]

    def __setitem__(self, idx: int, module: Union[Module, Callable]):
        self.layers[idx] = module

    def __rshift__(self, other):
        if isinstance(other, Module):
            self.layers.append(other)
        elif isinstance(other, Pipeline):
            self.layers.extend(other.layers)
        else:
            return NotImplemented
        return self

    def __irshift__(self, other):
        if isinstance(other, Module):
            self.layers.append(other)
        elif isinstance(other, Pipeline):
            self.layers.extend(other.layers)
        else:
            return NotImplemented
        return self

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x.realize()
        return x

    @property
    def params(self):
        params = []
        for layer in self.layers:
            if isinstance(layer, Module):
                params.extend(layer.params)
        return params

    @property
    def buffers(self):
        buffers = []
        for layer in self.layers:
            if isinstance(layer, Module):
                buffers.extend(layer.buffers)
        return buffers

    def freeze(self):
        for layer in self.layers:
            if isinstance(layer, Module):
                layer.freeze()

    def to(self, device: Device):
        for layer in self.layers:
            if isinstance(layer, Module):
                layer.to(device)
        return self

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, Module):
                layer.reset_parameters()
