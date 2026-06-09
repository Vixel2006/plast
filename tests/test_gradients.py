import numpy as np
import pytest
import plast


class TestAutogradBasics:
    def test_simple_chain(self, device, tol, rng):
        w = plast.tensor(rng.randn(2, 3).astype(np.float32), device=device, requires_grad=True)
        x = plast.tensor(rng.randn(3, 1).astype(np.float32), device=device, requires_grad=True)
        b = plast.tensor(rng.randn(2, 1).astype(np.float32), device=device, requires_grad=True)

        y = (w @ x + b).sum()
        plast.forward(y)
        y.backward()

        assert w.grad is not None
        assert x.grad is not None
        assert b.grad is not None

    def test_gradient_accumulation(self, device, tol, rng):
        x = plast.tensor(rng.randn(3).astype(np.float32), device=device, requires_grad=True)
        y = plast.tensor(rng.randn(3).astype(np.float32), device=device, requires_grad=True)

        z = (x * y).sum()
        plast.forward(z)
        z.backward()

        grad_x_first = x.grad.numpy().copy()

        plast.reset_transient_arenas()
        from plast.plast_core import zero_grad_cpu, Device

        try:
            from plast.plast_core import zero_grad_cuda
        except ImportError:
            zero_grad_cuda = None
        for p in [x, y]:
            raw = p._t
            if p.device == Device.CUDA:
                if zero_grad_cuda is not None:
                    zero_grad_cuda(raw)
            else:
                zero_grad_cpu(raw)

        z2 = (x * y).sum()
        plast.forward(z2)
        z2.backward()

        np.testing.assert_allclose(x.grad.numpy(), grad_x_first, **tol)

    def test_requires_grad_false(self, device, rng):
        x = plast.tensor(rng.randn(3).astype(np.float32), device=device, requires_grad=True)
        y = plast.tensor(rng.randn(3).astype(np.float32), device=device, requires_grad=False)

        z = (x + y).sum()
        plast.forward(z)
        z.backward()

        assert x.grad is not None
        assert y.grad is None or np.allclose(y.grad.numpy(), 0)

    def test_multi_input_graph(self, device, tol, rng):
        a = plast.tensor(rng.randn(3).astype(np.float32), device=device, requires_grad=True)
        b = plast.tensor(rng.randn(3).astype(np.float32), device=device, requires_grad=True)
        c = plast.tensor(rng.randn(3).astype(np.float32), device=device, requires_grad=True)

        d = a + b
        e = d * c
        loss = e.sum()
        plast.forward(loss)
        loss.backward()

        assert a.grad is not None
        assert b.grad is not None
        assert c.grad is not None

        np.testing.assert_allclose(a.grad.numpy(), c.numpy(), **tol)
        np.testing.assert_allclose(b.grad.numpy(), c.numpy(), **tol)


class TestGradientCorrectness:
    def test_add_grad(self, device, tol, rng):
        a = plast.tensor(rng.randn(2, 3).astype(np.float32), device=device, requires_grad=True)
        b = plast.tensor(rng.randn(2, 3).astype(np.float32), device=device, requires_grad=True)
        c = a + b
        loss = c.mean()
        plast.forward(loss)
        loss.backward()
        n = 6.0
        np.testing.assert_allclose(a.grad.numpy(), np.ones((2, 3)) / n, **tol)
        np.testing.assert_allclose(b.grad.numpy(), np.ones((2, 3)) / n, **tol)

    def test_sub_grad(self, device, tol, rng):
        a = plast.tensor(rng.randn(2, 3).astype(np.float32), device=device, requires_grad=True)
        b = plast.tensor(rng.randn(2, 3).astype(np.float32), device=device, requires_grad=True)
        c = a - b
        loss = c.sum()
        plast.forward(loss)
        loss.backward()
        np.testing.assert_allclose(a.grad.numpy(), np.ones((2, 3)), **tol)
        np.testing.assert_allclose(b.grad.numpy(), -np.ones((2, 3)), **tol)

    def test_div_grad(self, device, tol, rng):
        a_data = rng.randn(2, 3).astype(np.float32)
        b_data = np.abs(rng.randn(2, 3)).astype(np.float32) + 0.1
        a = plast.tensor(a_data, device=device, requires_grad=True)
        b = plast.tensor(b_data, device=device, requires_grad=True)
        c = a / b
        loss = c.sum()
        plast.forward(loss)
        loss.backward()
        np.testing.assert_allclose(a.grad.numpy(), 1.0 / b_data, **tol)
        np.testing.assert_allclose(b.grad.numpy(), -a_data / (b_data**2), **tol)

    def test_sin_grad(self, device, tol, rng):
        x_data = rng.randn(3).astype(np.float32)
        x = plast.tensor(x_data, device=device, requires_grad=True)
        y = x.sin()
        loss = y.sum()
        plast.forward(loss)
        loss.backward()
        np.testing.assert_allclose(x.grad.numpy(), np.cos(x_data), **tol)

    def test_cos_grad(self, device, tol, rng):
        x_data = rng.randn(3).astype(np.float32)
        x = plast.tensor(x_data, device=device, requires_grad=True)
        y = x.cos()
        loss = y.sum()
        plast.forward(loss)
        loss.backward()
        np.testing.assert_allclose(x.grad.numpy(), -np.sin(x_data), **tol)

    def test_exp_grad(self, device, tol, rng):
        x_data = rng.randn(3).astype(np.float32)
        x = plast.tensor(x_data, device=device, requires_grad=True)
        y = x.exp()
        loss = y.sum()
        plast.forward(loss)
        loss.backward()
        np.testing.assert_allclose(x.grad.numpy(), np.exp(x_data), **tol)

    def test_log_grad(self, device, tol, rng):
        x_data = (np.abs(rng.randn(3)) + 0.1).astype(np.float32)
        x = plast.tensor(x_data, device=device, requires_grad=True)
        y = x.log()
        loss = y.sum()
        plast.forward(loss)
        loss.backward()
        np.testing.assert_allclose(x.grad.numpy(), 1.0 / x_data, **tol)

    def test_neg_grad(self, device, tol, rng):
        x_data = rng.randn(3).astype(np.float32)
        x = plast.tensor(x_data, device=device, requires_grad=True)
        y = -x
        loss = y.sum()
        plast.forward(loss)
        loss.backward()
        np.testing.assert_allclose(x.grad.numpy(), -np.ones(3), **tol)

    def test_abs_grad(self, device, tol, rng):
        x_data = rng.randn(3).astype(np.float32)
        x = plast.tensor(x_data, device=device, requires_grad=True)
        y = abs(x)
        loss = y.sum()
        plast.forward(loss)
        loss.backward()
        expected = np.where(x_data > 0, 1.0, np.where(x_data < 0, -1.0, 0.0))
        np.testing.assert_allclose(x.grad.numpy(), expected, atol=1e-4)

    def test_tan_grad(self, device, tol, rng):
        x_data = rng.randn(3).astype(np.float32)
        x = plast.tensor(x_data, device=device, requires_grad=True)
        y = x.tan()
        loss = y.sum()
        plast.forward(loss)
        loss.backward()
        expected = 1.0 / (np.cos(x_data) ** 2)
        np.testing.assert_allclose(x.grad.numpy(), expected, **tol)

    def test_leaky_relu_grad(self, device, tol, rng):
        x_data = rng.randn(3).astype(np.float32)
        x = plast.tensor(x_data, device=device, requires_grad=True)
        y = plast.nn.functional.leaky_relu(x, 0.01)
        loss = y.sum()
        plast.forward(loss)
        loss.backward()
        expected = np.where(x_data > 0, 1.0, 0.01)
        np.testing.assert_allclose(x.grad.numpy(), expected, **tol)
