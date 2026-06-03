from .module import Module
from . import functional as F


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, input):
        return F.dropout(input, p=self.p, training=self.training)

    def __repr__(self):
        return f"Dropout(p={self.p})"
