"""plast.data — dataset and data loading utilities.

Provides:
- :class:`Dataset`       — abstract base class for datasets
- :class:`TensorDataset` — wraps in-memory tensors as a dataset
- :class:`DataLoader`    — batches and shuffles a Dataset

Example::

    import plast
    from plast.data import TensorDataset, DataLoader

    dataset = TensorDataset(X_train, y_train)
    loader  = DataLoader(dataset, batch_size=64, shuffle=True)

    for x, y in loader:
        ...
"""
from .dataset import Dataset, TensorDataset
from .dataloader import DataLoader
