import numpy as np
from .module import Module
from .._internal import Parameter, Device
from . import functional as F


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=Device.CPU):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weight parameter
        # Xavier initialization: uniform/normal distribution with bounds sqrt(2 / in_features)
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

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"
