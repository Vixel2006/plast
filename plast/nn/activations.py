from plast.core.tensor import Tensor
from plast.nn.module import Module


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

    def reset_parameters(self):
        pass


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return 2 / ((-2 * x).exp() + 1) - 1

    def reset_parameters(self):
        pass


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return 1 / ((-x).exp() + 1)

    def reset_parameters(self):
        pass
