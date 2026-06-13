import numpy as np
from ..tensor import Tensor
from ..plast_core import Device, DataLoader as _CDataLoader
from .._internal import tensor, get_arenas


class DataLoader:
    """Wraps a :class:`~plast.data.Dataset` and yields mini-batches.

    Args:
        dataset:    The dataset to load from.
        batch_size: Number of samples per batch (default 1).
        shuffle:    If ``True``, re-shuffle the data at every epoch (default
                    ``False``).
        drop_last:  If ``True``, drop the last incomplete batch (default
                    ``False``).
        device:     Target device for yielded tensors.  Defaults to
                    ``Device.CPU``.

    Example::

        loader = plast.data.DataLoader(dataset, batch_size=32, shuffle=True)
        for x, y in loader:
            ...

    Use ``len(loader)`` to get the number of batches.
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        device=None,
    ):
        if batch_size < 1:
            raise ValueError(
                f"batch_size must be a positive integer, got {batch_size}."
            )
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.device = device if device is not None else Device.CPU
        self._yielded_batches = []

        self.use_c_loader = hasattr(dataset, "_c_dataset")
        if self.use_c_loader:
            dev = device if device is not None else Device.CPU
            self._c_loader = _CDataLoader(
                dataset._c_dataset,
                batch_size,
                shuffle,
                drop_last,
                dev,
            )

    def __iter__(self):
        if self.use_c_loader:
            self._c_loader.reset()
            meta, data = get_arenas()
            while True:
                batch = self._c_loader.next_batch(meta, data)
                if batch is None:
                    break

                py_batch = [Tensor(t) for t in batch]
                if len(py_batch) == 1:
                    yield py_batch[0]
                else:
                    yield tuple(py_batch)
        else:
            import weakref
            from .._internal import reset_transient_arenas

            n = len(self.dataset)
            indices = np.arange(n)
            if self.shuffle:
                np.random.shuffle(indices)

            self._yielded_batches = []

            for start_idx in range(0, n, self.batch_size):
                if len(self._yielded_batches) > 1:
                    any_alive = False
                    for ref in self._yielded_batches[0]:
                        if ref() is not None:
                            any_alive = True
                            break
                    if not any_alive:
                        reset_transient_arenas()
                        self._yielded_batches = self._yielded_batches[1:]

                end_idx = start_idx + self.batch_size
                if end_idx > n:
                    if self.drop_last:
                        break
                    end_idx = n

                batch_indices = indices[start_idx:end_idx]

                try:
                    # Fast vectorized path: retrieve and process the entire batch at once
                    batch_samples = self.dataset[batch_indices]
                    if not isinstance(batch_samples, tuple):
                        batch_samples = (batch_samples,)

                    # Sanity check: confirm the first column has correct batch length
                    if len(batch_samples[0]) != len(batch_indices):
                        raise ValueError("Batch size mismatch in fast path")

                    batch_data = []
                    for col_data in batch_samples:
                        if isinstance(col_data, Tensor):
                            batch_data.append(col_data.to(self.device))
                        else:
                            arr = np.asarray(col_data, dtype=np.float32)
                            batch_data.append(tensor(arr, device=self.device))
                except Exception:
                    # Fallback to slow element-by-element path
                    samples = [self.dataset[i] for i in batch_indices]

                    # Support both tuple/list returns and single-value returns
                    if not isinstance(samples[0], (tuple, list)):
                        samples = [(s,) for s in samples]

                    num_outputs = len(samples[0])
                    batch_data = []
                    for col_idx in range(num_outputs):
                        col_samples = [s[col_idx] for s in samples]
                        col_np_list = []
                        for s in col_samples:
                            if isinstance(s, Tensor):
                                col_np_list.append(s.numpy())
                            elif isinstance(s, np.ndarray):
                                col_np_list.append(s.astype(np.float32))
                            else:
                                col_np_list.append(np.asarray(s, dtype=np.float32))
                        col_stacked = np.stack(col_np_list, axis=0)
                        batch_data.append(tensor(col_stacked, device=self.device))

                # Track weak references to yielded tensors
                batch_refs = []
                for t in batch_data:
                    if isinstance(t, Tensor):
                        batch_refs.append(weakref.ref(t))
                self._yielded_batches.append(batch_refs)

                if len(batch_data) == 1:
                    yield batch_data[0]
                else:
                    yield tuple(batch_data)

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __repr__(self) -> str:
        return (
            f"DataLoader(dataset={type(self.dataset).__name__}, "
            f"batch_size={self.batch_size}, shuffle={self.shuffle}, "
            f"drop_last={self.drop_last})"
        )
