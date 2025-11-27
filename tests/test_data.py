import pytest

from plast.data.dataset import Dataset


class ConcreteDataset(Dataset):
    """A concrete implementation of Dataset for testing purposes."""

    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class TestDataset:
    def test_abstract_methods_raise_error(self):
        # Instantiating Dataset directly should work, but calling its abstract methods should raise NotImplementedError
        dataset_instance = Dataset()
        with pytest.raises(NotImplementedError):
            _ = dataset_instance[0]
        with pytest.raises(NotImplementedError):
            _ = len(dataset_instance)

        # Test that concrete implementation works
        dataset = ConcreteDataset([1, 2, 3])
        assert len(dataset) == 3
        assert dataset[0] == 1
        assert dataset[1] == 2

    def test_concrete_dataset_getitem(self):
        dataset = ConcreteDataset([10, 20, 30])
        assert dataset[0] == 10
        assert dataset[1] == 20
        assert dataset[2] == 30
        with pytest.raises(IndexError):
            _ = dataset[3]

    def test_concrete_dataset_len(self):
        dataset = ConcreteDataset([1, 2, 3, 4, 5])
        assert len(dataset) == 5

        empty_dataset = ConcreteDataset([])
        assert len(empty_dataset) == 0
