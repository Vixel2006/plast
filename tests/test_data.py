import numpy as np
import pytest
import plast


class TestTensorDataset:
    def test_basic_iteration(self):
        x = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        y = np.array([0, 1, 0], dtype=np.float32)
        dataset = plast.data.TensorDataset(x, y)
        assert len(dataset) == 3
        for i in range(3):
            xi, yi = dataset[i]
            np.testing.assert_allclose(xi, x[i])
            assert yi == y[i]

    def test_single_tensor(self):
        x = np.array([1, 2, 3], dtype=np.float32)
        dataset = plast.data.TensorDataset(x)
        assert len(dataset) == 3

    def test_multiple_tensors(self):
        a = np.array([[1], [2]], dtype=np.float32)
        b = np.array([[3], [4]], dtype=np.float32)
        c = np.array([[5], [6]], dtype=np.float32)
        dataset = plast.data.TensorDataset(a, b, c)
        a0, b0, c0 = dataset[0]
        assert a0 == 1
        assert b0 == 3
        assert c0 == 5


class TestDataLoader:
    def test_basic_batching(self, device):
        n = 10
        x = np.arange(n * 2, dtype=np.float32).reshape(n, 2)
        y = np.arange(n, dtype=np.float32).reshape(n, 1)
        dataset = plast.data.TensorDataset(x, y)
        loader = plast.data.DataLoader(dataset, batch_size=4, shuffle=False, device=device)

        batches = list(loader)
        assert len(batches) == 3
        x0, y0 = batches[0]
        assert x0.shape == [4, 2]
        assert y0.shape == [4, 1]

    def test_shuffle(self, device):
        n = 20
        x = np.arange(n, dtype=np.float32).reshape(n, 1)
        y = np.arange(n, dtype=np.float32).reshape(n, 1)
        dataset = plast.data.TensorDataset(x, y)
        loader = plast.data.DataLoader(dataset, batch_size=5, shuffle=True, device=device)

        first_batch_x = next(iter(loader))[0].numpy().flatten()
        loader2 = plast.data.DataLoader(
            plast.data.TensorDataset(
                np.arange(n, dtype=np.float32).reshape(n, 1),
                np.arange(n, dtype=np.float32).reshape(n, 1),
            ),
            batch_size=5, shuffle=True, device=device,
        )
        second_first = next(iter(loader2))[0].numpy().flatten()

        assert len(first_batch_x) == 5

    def test_drop_last(self, device):
        n = 10
        dataset = plast.data.TensorDataset(
            np.arange(n, dtype=np.float32).reshape(n, 1),
            np.arange(n, dtype=np.float32).reshape(n, 1),
        )
        loader = plast.data.DataLoader(
            dataset, batch_size=3, shuffle=False, drop_last=True, device=device,
        )
        batches = list(loader)
        assert len(batches) == 3

    def test_no_drop_last(self, device):
        n = 10
        dataset = plast.data.TensorDataset(
            np.arange(n, dtype=np.float32).reshape(n, 1),
            np.arange(n, dtype=np.float32).reshape(n, 1),
        )
        loader = plast.data.DataLoader(
            dataset, batch_size=3, shuffle=False, drop_last=False, device=device,
        )
        batches = list(loader)
        assert len(batches) == 4

    def test_multi_tensor_dataloader(self, device):
        n = 6
        a = np.arange(n * 2, dtype=np.float32).reshape(n, 2)
        b = np.arange(n, dtype=np.float32).reshape(n, 1)
        c = np.ones(n, dtype=np.float32)
        dataset = plast.data.TensorDataset(a, b, c)
        loader = plast.data.DataLoader(dataset, batch_size=2, shuffle=False, device=device)

        for batch in loader:
            assert len(batch) == 3
            assert batch[0].shape == [2, 2]
            assert batch[1].shape == [2, 1]
            assert batch[2].shape == [2]

    def test_device_transfer(self):
        x = np.array([[1, 2]], dtype=np.float32)
        y = np.array([[0]], dtype=np.float32)
        dataset = plast.data.TensorDataset(x, y)
        loader = plast.data.DataLoader(
            dataset, batch_size=1, shuffle=False, device=plast.Device.CPU,
        )
        batch_x, batch_y = next(iter(loader))
        assert batch_x.device == plast.Device.CPU

    def test_large_batch(self, device):
        n = 128
        x = np.random.randn(n, 32).astype(np.float32)
        y = np.random.randint(0, 2, (n, 1)).astype(np.float32)
        dataset = plast.data.TensorDataset(x, y)
        loader = plast.data.DataLoader(dataset, batch_size=32, shuffle=True, device=device)
        batches = list(loader)
        assert len(batches) == 4
        total = sum(batch[0].shape[0] for batch in batches)
        assert total == n

    def test_epoch_iteration_consistency(self, device):
        n = 8
        x = np.arange(n, dtype=np.float32).reshape(n, 1)
        y = np.arange(n, dtype=np.float32).reshape(n, 1)
        dataset = plast.data.TensorDataset(x, y)
        loader = plast.data.DataLoader(
            dataset, batch_size=4, shuffle=False, device=device,
        )
        epoch1 = [batch[0].numpy() for batch in loader]
        epoch2 = [batch[0].numpy() for batch in loader]
        np.testing.assert_allclose(
            np.concatenate(epoch1), np.concatenate(epoch2),
        )
