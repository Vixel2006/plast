from abc import ABC, abstractmethod
from typing import Any, Iterable

from plast.core.tensor import Tensor


class Module(ABC):
    @abstractmethod
    def forward(self, x, *args, **kwargs):
        pass

    def __call__(self, x, *args: Any, **kwds: Any) -> Any:
        return self.forward(x, *args, **kwds)

    def __rshift__(self, other):
        from .pipeline import Pipeline

        if isinstance(other, Module):
            return Pipeline(self, other)
        elif isinstance(other, Pipeline):
            new_pipeline = Pipeline(self)
            new_pipeline.layers.extend(other.layers)
            return new_pipeline
        else:
            return NotImplemented

    @property
    def params(self):
        params = []
        for elem in self.__dict__.values():
            if isinstance(
                elem, Tensor
            ):  # Assuming Tensor now has requires_grad property
                if hasattr(elem, "requires_grad") and elem.requires_grad:
                    params.append(elem)
            elif isinstance(elem, Module):
                params.extend(elem.params)
        return params

    @property
    def buffers(self):
        buffers = []
        for elem in self.__dict__.values():
            if isinstance(
                elem, Tensor
            ):  # Assuming Tensor now has requires_grad property
                if hasattr(elem, "requires_grad") and not elem.requires_grad:
                    buffers.append(elem)
            elif isinstance(elem, Module):
                buffers.extend(elem.buffers)
        return buffers

    def freeze(self):
        for param in self.params:
            if hasattr(param, "requires_grad"):
                param.requires_grad = False

    def to(
        self, device: str
    ):  # Changed Device to str for now, as Device class is removed
        for name, value in self.__dict__.items():
            if isinstance(value, Tensor):
                setattr(self, name, value.to(device))
            elif isinstance(value, Module):
                setattr(self, name, value.to(device))
        return self

    def reset_parameters(self):
        pass
