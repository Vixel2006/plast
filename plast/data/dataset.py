import numpy as np


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
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(t[index] for t in self.tensors)

    def __len__(self) -> int:
        return len(self.tensors[0])

    def __repr__(self) -> str:
        shapes = []
        for t in self.tensors:
            if hasattr(t, "shape"):
                shapes.append(str(list(t.shape)) if hasattr(t.shape, "__iter__") else str(t.shape))
            else:
                shapes.append(f"[{len(t)}, ...]")
        return f"TensorDataset(tensors=[{', '.join(shapes)}], size={len(self)})"
