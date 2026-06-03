import numpy as np
import pytest
import plast


class TestSGD:
    def test_sgd_step(self, device, tol, rng):
        w = plast.tensor(rng.randn(2, 3).astype(np.float32), device=device, requires_grad=True)
        x = plast.tensor(rng.randn(3, 1).astype(np.float32), device=device, requires_grad=True)

        y = (w @ x).sum()
        plast.forward(y)
        y.backward()

        old_w = w.numpy().copy()
        lr = 0.1
        optim = plast.optim.SGD([w], lr=lr)
        optim.step()
        expected = old_w - lr * w.grad.numpy()
        np.testing.assert_allclose(w.numpy(), expected, **tol)

    def test_sgd_zero_grad(self, device, rng):
        w = plast.tensor(rng.randn(2, 3).astype(np.float32), device=device, requires_grad=True)
        x = plast.tensor(rng.randn(3, 1).astype(np.float32), device=device, requires_grad=True)

        y = (w @ x).sum()
        plast.forward(y)
        y.backward()

        optim = plast.optim.SGD([w], lr=0.01)
        optim.zero_grad()
        for p in optim.param_groups[0]["params"]:
            if p.grad is not None:
                assert np.allclose(p.grad.numpy(), 0)

    def test_sgd_multiple_params(self, device, tol, rng):
        w1 = plast.tensor(rng.randn(3).astype(np.float32), device=device, requires_grad=True)
        w2 = plast.tensor(rng.randn(3).astype(np.float32), device=device, requires_grad=True)
        x = plast.tensor(rng.randn(3).astype(np.float32), device=device)

        y = (w1 * x * w2).sum()
        plast.forward(y)
        y.backward()

        old = [w1.numpy().copy(), w2.numpy().copy()]
        grads = [w1.grad.numpy().copy(), w2.grad.numpy().copy()]

        lr = 0.05
        optim = plast.optim.SGD([w1, w2], lr=lr)
        optim.step()

        np.testing.assert_allclose(w1.numpy(), old[0] - lr * grads[0], **tol)
        np.testing.assert_allclose(w2.numpy(), old[1] - lr * grads[1], **tol)


class TestAdam:
    def test_adam_step(self, device, tol, rng):
        w = plast.tensor(rng.randn(2, 3).astype(np.float32), device=device, requires_grad=True)
        x = plast.tensor(rng.randn(3, 1).astype(np.float32), device=device)

        y = (w @ x).sum()
        plast.forward(y)
        y.backward()

        old_w = w.numpy().copy()
        lr = 0.01
        optim = plast.optim.Adam([w], lr=lr)
        optim.step()
        assert not np.allclose(w.numpy(), old_w)

    def test_adam_multiple_steps(self, device, rng):
        w = plast.tensor(rng.randn(3).astype(np.float32), device=device, requires_grad=True)
        x = plast.tensor(np.ones(3, dtype=np.float32), device=device)

        optim = plast.optim.Adam([w], lr=0.01)

        for _ in range(5):
            optim.zero_grad()
            y = (w * x).sum()
            plast.forward(y)
            y.backward()
            optim.step()
            plast.reset_transient_arenas()

        assert w.grad is not None


class TestAdamW:
    def test_adamw_step(self, device, rng):
        w = plast.tensor(rng.randn(2, 3).astype(np.float32), device=device, requires_grad=True)
        x = plast.tensor(rng.randn(3, 1).astype(np.float32), device=device)

        y = (w @ x).sum()
        plast.forward(y)
        y.backward()

        old_w = w.numpy().copy()
        optim = plast.optim.AdamW([w], lr=0.01, weight_decay=0.01)
        optim.step()

        assert not np.allclose(w.numpy(), old_w)

    def test_adamw_weight_decay(self, device, tol, rng):
        w = plast.tensor(np.array([2.0, 2.0], dtype=np.float32), device=device, requires_grad=True)
        y = (w * w).sum()
        plast.forward(y)
        y.backward()

        optim = plast.optim.AdamW([w], lr=0.1, weight_decay=0.1)
        old_w = w.numpy().copy()

        optim.step()
        assert not np.allclose(w.numpy(), old_w)


class TestLRSchedulers:
    def test_step_lr(self, device, rng):
        w = plast.tensor(rng.randn(3).astype(np.float32), device=device, requires_grad=True)
        optim = plast.optim.SGD([w], lr=0.1)
        scheduler = plast.optim.StepLR(optim, step_size=2, gamma=0.5)

        initial_lr = optim.param_groups[0]["lr"]
        scheduler.step()
        scheduler.step()
        current_lr = optim.param_groups[0]["lr"]
        assert current_lr == initial_lr * 0.5

    def test_multi_step_lr(self, device, rng):
        w = plast.tensor(rng.randn(3).astype(np.float32), device=device, requires_grad=True)
        optim = plast.optim.SGD([w], lr=0.1)
        scheduler = plast.optim.MultiStepLR(optim, milestones=[1, 3], gamma=0.5)

        scheduler.step()
        assert optim.param_groups[0]["lr"] == 0.05
        scheduler.step()
        assert optim.param_groups[0]["lr"] == 0.05
        scheduler.step()
        assert optim.param_groups[0]["lr"] == 0.025

    def test_exponential_lr(self, device, rng):
        w = plast.tensor(rng.randn(3).astype(np.float32), device=device, requires_grad=True)
        optim = plast.optim.SGD([w], lr=0.1)
        scheduler = plast.optim.ExponentialLR(optim, gamma=0.9)

        scheduler.step()
        assert optim.param_groups[0]["lr"] == pytest.approx(0.09)
        scheduler.step()
        assert optim.param_groups[0]["lr"] == pytest.approx(0.081)

    def test_cosine_annealing_lr(self, device, rng):
        w = plast.tensor(rng.randn(3).astype(np.float32), device=device, requires_grad=True)
        optim = plast.optim.SGD([w], lr=0.1)
        scheduler = plast.optim.CosineAnnealingLR(optim, T_max=4)

        scheduler.step()
        assert optim.param_groups[0]["lr"] > 0

    def test_reduce_lr_on_plateau(self, device, rng):
        w = plast.tensor(rng.randn(3).astype(np.float32), device=device, requires_grad=True)
        optim = plast.optim.SGD([w], lr=0.1)
        scheduler = plast.optim.ReduceLROnPlateau(optim, patience=2, factor=0.5)

        for _ in range(5):
            scheduler.step(1.0)

        assert optim.param_groups[0]["lr"] < 0.1
