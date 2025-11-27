from abc import ABC, abstractmethod

class Dataset(ABC):
    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
