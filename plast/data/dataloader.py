import numpy as np
from ..plast_core import Tensor, Device
from .._internal import tensor


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False, device=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.device = device

    def __iter__(self):
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
