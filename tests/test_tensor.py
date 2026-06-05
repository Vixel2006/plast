import numpy as np
import pytest
import plast


class TestTensorCreation:
    def test_tensor_from_numpy(self, device):
        data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        t = plast.tensor(data, device=device)
        assert t.shape == [2, 2]
        assert t.ndim == 2
        assert t.device == device
        assert t.dtype == plast.DType.Float32
        assert not t.requires_grad

    def test_tensor_requires_grad(self, device):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        t = plast.tensor(data, device=device, requires_grad=True)
        assert t.requires_grad

    def test_tensor_zeros(self, device):
        t = plast.tensor(np.zeros((3, 4), dtype=np.float32), device=device)
        np.testing.assert_allclose(t.numpy(), np.zeros((3, 4)), atol=1e-6)

    def test_tensor_ones(self, device):
        t = plast.tensor(np.ones((2, 3), dtype=np.float32), device=device)
        np.testing.assert_allclose(t.numpy(), np.ones((2, 3)), atol=1e-6)

    def test_tensor_1d(self, device):
        t = plast.tensor([1, 2, 3], device=device)
        assert t.shape == [3]
        np.testing.assert_allclose(t.numpy(), [1, 2, 3])

    def test_tensor_3d(self, device):
        data = np.random.randn(2, 3, 4).astype(np.float32)
        t = plast.tensor(data, device=device)
        assert t.shape == [2, 3, 4]
        assert t.ndim == 3


class TestTensorProperties:
    def test_numel(self, device):
        t = plast.tensor(np.zeros((3, 4, 5), dtype=np.float32), device=device)
        assert t.numel() == 60

    def test_is_contiguous(self, device):
        t = plast.tensor(np.zeros((4, 4), dtype=np.float32), device=device)
        assert t.is_contiguous

    def test_strides(self, device):
        t = plast.tensor(np.zeros((4, 4), dtype=np.float32), device=device)
        strides = t.strides
        assert len(strides) == 2
        assert strides[0] >= strides[1]

    def test_repr(self, device):
        t = plast.tensor([1.0, 2.0], device=device)
        s = repr(t)
        assert "tensor" in s

    def test_device_transfer(self, device):
        t = plast.tensor([1, 2, 3], device=plast.Device.CPU)
        if device == plast.Device.CUDA:
            t_gpu = t.to(plast.Device.CUDA)
            assert t_gpu.device == plast.Device.CUDA
            np.testing.assert_allclose(t_gpu.numpy(), [1, 2, 3])
            t_back = t_gpu.to(plast.Device.CPU)
            assert t_back.device == plast.Device.CPU


class TestTensorConversions:
    def test_numpy_roundtrip(self, device):
        data = np.random.randn(3, 4).astype(np.float32)
        t = plast.tensor(data, device=device)
        out = t.numpy()
        np.testing.assert_allclose(out, data, atol=1e-6)

    def test_item_single(self, device):
        t = plast.tensor(np.array([42.0], dtype=np.float32), device=device)
        assert t.item() == 42.0

    def test_item_multi_raises(self, device):
        t = plast.tensor([1, 2, 3], device=device)
        with pytest.raises(ValueError):
            t.item()

    def test_copy_from_numpy(self, device):
        t = plast.tensor(np.zeros((2, 2), dtype=np.float32), device=device)
        new_data = np.array([[5, 6], [7, 8]], dtype=np.float32)
        t.copy_from_numpy(new_data)
        np.testing.assert_allclose(t.numpy(), new_data, atol=1e-6)
