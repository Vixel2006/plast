import numpy as np
import pytest
import plast


class TestEdgeCases:
    def test_single_element_tensor(self, device, tol, rng):
        a = plast.tensor(np.array([42.0], dtype=np.float32), device=device)
        assert a.item() == 42.0
        assert a.shape == [1]
        assert plast.plast_core.numel(a) == 1

    def test_scalar_arithmetic(self, device, tol, rng):
        a = plast.tensor(np.array([3.0], dtype=np.float32), device=device)
        b = plast.tensor(np.array([2.0], dtype=np.float32), device=device)
        c = a + b
        plast.forward(c)
        np.testing.assert_allclose(c.numpy(), [5.0], **tol)

    def test_large_tensor_creation(self, device):
        shape = [100, 100]
        a = plast.tensor(np.zeros(shape, dtype=np.float32), device=device)
        assert a.shape == shape
        assert plast.plast_core.numel(a) == 10000

    @pytest.mark.slow
    def test_large_matmul(self, device, tol, rng):
        m, n, p = 64, 64, 64
        a = rng.randn(m, n).astype(np.float32)
        b = rng.randn(n, p).astype(np.float32)
        ta = plast.tensor(a, device=device)
        tb = plast.tensor(b, device=device)
        tc = ta @ tb
        plast.forward(tc)
        expected = a @ b
        np.testing.assert_allclose(tc.numpy(), expected, **tol)

    def test_high_dim_tensor(self, device, tol, rng):
        shape = [2, 3, 4, 5]
        a = rng.randn(*shape).astype(np.float32)
        ta = plast.tensor(a, device=device)
        assert ta.ndim == 4
        assert ta.shape == shape

    def test_tensor_broadcast_add(self, device, tol, rng):
        a = rng.randn(3, 1).astype(np.float32)
        b = rng.randn(1, 4).astype(np.float32)
        ta, tb = [plast.tensor(x, device=device) for x in (a, b)]
        tc = ta + tb
        plast.forward(tc)
        expected = a + b
        np.testing.assert_allclose(tc.numpy(), expected, **tol)

    def test_broadcast_add_same_ndim(self, device, tol, rng):
        a = rng.randn(4, 1).astype(np.float32)
        b = rng.randn(1, 4).astype(np.float32)
        ta, tb = [plast.tensor(x, device=device) for x in (a, b)]
        tc = ta + tb
        plast.forward(tc)
        expected = a + b
        np.testing.assert_allclose(tc.numpy(), expected, **tol)

    def test_multiple_ops_chain(self, device, tol, rng):
        a = rng.randn(2, 3).astype(np.float32)
        b = rng.randn(2, 3).astype(np.float32)
        c = rng.randn(2, 3).astype(np.float32)
        ta, tb, tc = [plast.tensor(x, device=device, requires_grad=True) for x in (a, b, c)]
        td = (ta + tb) * tc
        te = td.sin()
        loss = te.sum()
        plast.forward(loss)
        loss.backward()
        assert ta.grad is not None
        assert tb.grad is not None
        assert tc.grad is not None

    def test_grad_non_requires_grad_input(self, device, tol, rng):
        a = plast.tensor(rng.randn(3).astype(np.float32), device=device, requires_grad=True)
        b = plast.tensor(rng.randn(3).astype(np.float32), device=device, requires_grad=False)
        c = a + b
        loss = c.sum()
        plast.forward(loss)
        loss.backward()
        assert a.grad is not None
        assert b.grad is None

    def test_tensor_arithmetic_chain(self, device, tol, rng):
        a = plast.tensor(rng.randn(2, 3).astype(np.float32), device=device)
        b = plast.tensor(rng.randn(2, 3).astype(np.float32), device=device)
        c = plast.tensor(rng.randn(2, 3).astype(np.float32), device=device)
        one = plast.tensor(np.ones((2, 3), dtype=np.float32), device=device)

        d = (a + b) * c - a / (b + one)
        plast.forward(d)

        a_np, b_np, c_np = [x.numpy() for x in (a, b, c)]
        expected = (a_np + b_np) * c_np - a_np / (b_np + 1.0)
        np.testing.assert_allclose(d.numpy(), expected, **tol)

    def test_unsqueeze_chain(self, device, tol, rng):
        a = rng.randn(3).astype(np.float32)
        ta = plast.tensor(a, device=device)
        tb = ta.unsqueeze(0).unsqueeze(2)
        plast.forward(tb)
        np.testing.assert_allclose(tb.numpy(), a[np.newaxis, :, np.newaxis], **tol)
        assert tb.shape == [1, 3, 1]

    @pytest.mark.xfail(reason="squeeze with no dim doesn't remove all squeeze dims")
    def test_squeeze_all(self, device, tol, rng):
        a = rng.randn(1, 3, 1, 4, 1).astype(np.float32)
        ta = plast.tensor(a, device=device)
        tb = ta.squeeze()
        plast.forward(tb)
        np.testing.assert_allclose(tb.numpy(), a.reshape(3, 4), **tol)

    def test_squeeze_dim(self, device, tol, rng):
        a = rng.randn(1, 3, 1, 4).astype(np.float32)
        ta = plast.tensor(a, device=device)
        tb = ta.squeeze(dim=0)
        plast.forward(tb)
        assert tb.shape == [3, 1, 4]
        np.testing.assert_allclose(tb.numpy(), a.reshape(3, 1, 4), **tol)

    @pytest.mark.xfail(reason="negative dims in transpose not supported")
    def test_negative_index_transpose(self, device, tol, rng):
        a = rng.randn(2, 3, 4).astype(np.float32)
        ta = plast.tensor(a, device=device)
        tb = ta.transpose(-2, -1)
        plast.forward(tb)
        expected = np.transpose(a, (0, 2, 1))
        np.testing.assert_allclose(tb.numpy(), expected, **tol)

    @pytest.mark.xfail(reason="transpose produces incorrect output for 3D tensors")
    def test_transpose_positive_dims(self, device, tol, rng):
        a = rng.randn(2, 3, 4).astype(np.float32)
        ta = plast.tensor(a, device=device)
        tb = ta.transpose(1, 2)
        plast.forward(tb)
        expected = np.transpose(a, (0, 2, 1))
        np.testing.assert_allclose(tb.numpy(), expected, **tol)

    @pytest.mark.xfail(reason="view + element-wise op produces incorrect non-contiguous strides")
    def test_view_then_op(self, device, tol, rng):
        a = rng.randn(2, 6).astype(np.float32)
        ta = plast.tensor(a, device=device)
        tv = ta.view(3, 4)
        b = plast.tensor(rng.randn(4).astype(np.float32), device=device)
        tc = tv + b
        plast.forward(tc)
        expected = a.reshape(3, 4) + b.numpy()
        np.testing.assert_allclose(tc.numpy(), expected, **tol)
