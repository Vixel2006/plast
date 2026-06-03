from .plast_core import Device, DType, OpType, forward, backward
from ._internal import init_arenas, tensor, Parameter, get_arenas, get_persistent_arenas, reset_transient_arenas

# Import tensor_ops to apply monkeypatching to the pybind11 Tensor class
from . import tensor_ops

# Expose subpackages
from . import nn
from . import optim
from . import data
from . import experiment
