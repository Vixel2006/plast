import numpy as np
from ..tensor import Tensor
from .._internal import tensor
from ..plast_core import TensorDataset as _CTensorDataset


class Dataset:
    """Abstract base class for all datasets.

    Subclasses must implement :meth:`__getitem__` and :meth:`__len__`::

        class MyDataset(plast.data.Dataset):
            def __init__(self, X, y):
                self.X = X
                self.y = y

            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]

            def __len__(self):
                return len(self.X)
    """

    def __getitem__(self, index):
        raise NotImplementedError(
            f"{type(self).__name__} must implement __getitem__(index)."
        )

    def __len__(self):
        raise NotImplementedError(
            f"{type(self).__name__} must implement __len__()."
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={len(self)})"


class TensorDataset(Dataset):
    """Wraps one or more tensors (or numpy arrays) as a dataset.

    Each call to ``__getitem__`` returns a tuple of the *i*-th element
    from each tensor::

        dataset = plast.data.TensorDataset(X, y)
        x_i, y_i = dataset[0]

    All tensors must have the same size in the first dimension.
    """

    def __init__(self, *tensors):
        if not tensors:
            raise ValueError("TensorDataset requires at least one tensor.")
        sizes = []
        for i, t in enumerate(tensors):
            if hasattr(t, "__len__"):
                sizes.append(len(t))
            else:
                raise TypeError(
                    f"TensorDataset: argument {i} has no len(); "
                    f"got {type(t).__name__}."
                )
        if len(set(sizes)) > 1:
            raise ValueError(
                f"TensorDataset: all tensors must have the same first-dimension size, "
                f"got sizes {sizes}."
            )

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

    def __len__(self) -> int:
        return len(self._c_dataset)

    def __repr__(self) -> str:
        shapes = []
        for t in self.tensors:
            shapes.append(str(list(t.shape)))
        return f"TensorDataset(tensors=[{', '.join(shapes)}], size={len(self)})"
