from ..tensor import Tensor
from .._internal import tensor
from ..plast_core import TensorDataset as _CTensorDataset

class Dataset:
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class TensorDataset(Dataset):
    def __init__(self, *tensors):
        self.raw_tensors = tensors
        self.tensors = []
        for t in tensors:
            if isinstance(t, Tensor):
                self.tensors.append(t)
            else:
                # We convert numpy arrays to persistent CPU Tensors.
                self.tensors.append(tensor(t, persistent=True))
        
        c_tensors = [t._t for t in self.tensors]
        self._c_dataset = _CTensorDataset(c_tensors)

    def __getitem__(self, index):
        return tuple(t[index] for t in self.raw_tensors)

    def __len__(self):
        return len(self._c_dataset)

