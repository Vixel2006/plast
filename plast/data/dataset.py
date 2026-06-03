class Dataset:
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class TensorDataset(Dataset):
    def __init__(self, *tensors):
        # We accept numpy arrays or Plast Tensors.
        # We store their data (either as numpy arrays or Tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(t[index] for t in self.tensors)

    def __len__(self):
        return len(self.tensors[0])
