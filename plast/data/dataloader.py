import numpy as np
from ..tensor import Tensor
from ..plast_core import Device, DataLoader as _CDataLoader
from .._internal import tensor, get_arenas


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False, device=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.device = device

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
            n = len(self.dataset)
            indices = np.arange(n)
            if self.shuffle:
                np.random.shuffle(indices)

            device = self.device
            if device is None:
                device = Device.CPU

            for start_idx in range(0, n, self.batch_size):
                end_idx = start_idx + self.batch_size
                if end_idx > n:
                    if self.drop_last:
                        break
                    end_idx = n

                batch_indices = indices[start_idx:end_idx]

                samples = [self.dataset[i] for i in batch_indices]
                num_outputs = len(samples[0])
                batch_data = []
                for col_idx in range(num_outputs):
                    col_samples = [s[col_idx] for s in samples]
                    col_np = []
                    for s in col_samples:
                        if isinstance(s, Tensor):
                            col_np.append(s.numpy())
                        else:
                            col_np.append(s)
                    col_stacked = np.stack(col_np, axis=0)
                    batch_data.append(tensor(col_stacked, device=device))

                if len(batch_data) == 1:
                    yield batch_data[0]
                else:
                    yield tuple(batch_data)

    def __len__(self):
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
