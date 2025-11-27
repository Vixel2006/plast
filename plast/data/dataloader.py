from __future__ import annotations

import math
import random
from typing import Iterator, List, Optional, Sized

import numpy as np

from plast.functions import from_data


class DataLoader:
    def __init__(self, dataset: Sized, batch_size: int, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))

    def __iter__(self) -> Iterator[List]:
        if self.shuffle:
            random.shuffle(self.indices)

        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i : i + self.batch_size]
            batch_data = [self.dataset[idx] for idx in batch_indices]

            transposed_batch = list(zip(*batch_data))

            stacked_tensors = []
            for items in transposed_batch:
                numpy_arrays = []
                for item in items:
                    data = item.data
                    if data.shape[0] == 1 and len(data.shape) > 1:
                        data = np.squeeze(data, axis=0)
                    numpy_arrays.append(data)
                stacked_numpy_array = np.stack(numpy_arrays, axis=0)

                first_tensor = items[0]
                stacked_tensor_shape = stacked_numpy_array.shape
                stacked_tensor_device = first_tensor.device
                stacked_tensor_requires_grad = first_tensor.requires_grad

                stacked_tensors.append(
                    from_data(
                        stacked_tensor_shape,
                        stacked_numpy_array,
                        device=stacked_tensor_device,
                        requires_grad=stacked_tensor_requires_grad,
                    )
                )

            yield tuple(stacked_tensors)

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)
