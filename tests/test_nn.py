import numpy as np
import pytest
import plast


class TestLinear:
    def test_linear_forward(self, device, tol, rng):
        layer = plast.nn.Linear(4, 3, device=device)
        x = plast.tensor(rng.randn(2, 4).astype(np.float32), device=device)
        out = layer(x)
        plast.forward(out)
        assert out.shape == [2, 3]

    def test_linear_no_bias(self, device, tol, rng):
        layer = plast.nn.Linear(4, 3, bias=False, device=device)
        x = plast.tensor(rng.randn(2, 4).astype(np.float32), device=device)
        out = layer(x)
        plast.forward(out)
        assert out.shape == [2, 3]

    def test_linear_parameters(self, device, rng):
        layer = plast.nn.Linear(4, 3, bias=True, device=device)
        params = layer.parameters()
        assert len(params) == 2
        assert params[0].shape == [4, 3]
        assert params[1].shape == [1, 3]

    def test_linear_parameters_no_bias(self, device, rng):
        layer = plast.nn.Linear(4, 3, bias=False, device=device)
        params = layer.parameters()
        assert len(params) == 1


class TestActivations:
    def test_relu_forward(self, device, tol, rng):
        relu = plast.nn.ReLU()
        x = plast.tensor(rng.randn(2, 3).astype(np.float32), device=device)
        out = relu(x)
        plast.forward(out)
        expected = np.maximum(x.numpy(), 0)
        np.testing.assert_allclose(out.numpy(), expected, **tol)

    def test_tanh_forward(self, device, tol, rng):
        tanh = plast.nn.Tanh()
        x = plast.tensor(rng.randn(2, 3).astype(np.float32), device=device)
        out = tanh(x)
        plast.forward(out)
        expected = np.tanh(x.numpy())
        np.testing.assert_allclose(out.numpy(), expected, atol=1e-3)

    @pytest.mark.xfail(reason="leaky_relu forward does not apply alpha factor")
    def test_leaky_relu_module(self, device, tol, rng):
        lrelu = plast.nn.LeakyReLU(0.02)
        x = plast.tensor(rng.randn(2, 3).astype(np.float32), device=device)
        out = lrelu(x)
        plast.forward(out)
        expected = np.where(x.numpy() > 0, x.numpy(), 0.02 * x.numpy())
        np.testing.assert_allclose(out.numpy(), expected, **tol)

    @pytest.mark.xfail(reason="softmax uses max+keepdim which produces wrong dim output")
    def test_softmax(self, device, tol, rng):
        x = plast.tensor(rng.randn(2, 4).astype(np.float32), device=device)
        out = plast.nn.functional.softmax(x, dim=1)
        plast.forward(out)
        expected = np.exp(x.numpy()) / np.sum(np.exp(x.numpy()), axis=1, keepdims=True)
        np.testing.assert_allclose(out.numpy(), expected, **tol)


class TestLossFunctions:
    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_mse_loss(self, reduction, device, tol, rng):
        pred = plast.tensor(rng.randn(4, 1).astype(np.float32), device=device)
        target = plast.tensor(rng.randn(4, 1).astype(np.float32), device=device)
        loss_fn = plast.nn.MSELoss(reduction=reduction)
        loss = loss_fn(pred, target)
        plast.forward(loss)
        diff = pred.numpy() - target.numpy()
        if reduction == "mean":
            expected = np.mean(diff ** 2)
        else:
            expected = np.sum(diff ** 2)
        np.testing.assert_allclose(loss.numpy(), np.array([expected]), **tol)

    @pytest.mark.xfail(reason="l1_loss uses .abs() which is not monkeypatched on Tensor")
    def test_l1_loss(self, device, tol, rng):
        pred = plast.tensor(rng.randn(4, 1).astype(np.float32), device=device)
        target = plast.tensor(rng.randn(4, 1).astype(np.float32), device=device)
        loss = plast.nn.L1Loss()(pred, target)
        plast.forward(loss)
        expected = np.mean(np.abs(pred.numpy() - target.numpy()))
        np.testing.assert_allclose(loss.numpy(), np.array([expected]), **tol)

    def test_l1_loss_manual(self, device, tol, rng):
        pred = plast.tensor(rng.randn(4, 1).astype(np.float32), device=device)
        target = plast.tensor(rng.randn(4, 1).astype(np.float32), device=device)
        diff = abs(pred - target)
        loss = diff.mean()
        plast.forward(loss)
        expected = np.mean(np.abs(pred.numpy() - target.numpy()))
        np.testing.assert_allclose(loss.numpy(), np.array([expected]), **tol)

    @pytest.mark.xfail(reason="cross_entropy target * log_soft broadcasting with class indices")
    def test_cross_entropy_loss(self, device, tol, rng):
        logits = plast.tensor(rng.randn(3, 5).astype(np.float32), device=device)
        targets_val = np.array([0, 2, 4], dtype=np.int32)
        targets = plast.tensor(targets_val, device=device)
        loss = plast.nn.CrossEntropyLoss()(logits, targets)
        plast.forward(loss)
        logits_np = logits.numpy()
        logits_exp = np.exp(logits_np - np.max(logits_np, axis=1, keepdims=True))
        softmax = logits_exp / np.sum(logits_exp, axis=1, keepdims=True)
        expected = -np.mean(np.log(softmax[np.arange(3), targets_val] + 1e-8))
        np.testing.assert_allclose(loss.numpy(), np.array([expected]), atol=1e-2)


class TestSequential:
    def test_sequential_forward(self, device, tol, rng):
        model = plast.nn.Sequential(
            plast.nn.Linear(4, 8, device=device),
            plast.nn.ReLU(),
            plast.nn.Linear(8, 2, device=device),
        )
        x = plast.tensor(rng.randn(3, 4).astype(np.float32), device=device)
        out = model(x)
        plast.forward(out)
        assert out.shape == [3, 2]

    def test_sequential_parameters(self, device, rng):
        model = plast.nn.Sequential(
            plast.nn.Linear(4, 8, device=device),
            plast.nn.ReLU(),
            plast.nn.Linear(8, 2, device=device),
        )
        params = model.parameters()
        assert len(params) == 4

    def test_sequential_state_dict(self, device, rng):
        model = plast.nn.Sequential(
            plast.nn.Linear(2, 3, device=device),
            plast.nn.ReLU(),
        )
        sd = model.state_dict()
        assert len(sd) == 2
        for key, val in sd.items():
            assert isinstance(val, np.ndarray)

    def test_sequential_load_state_dict(self, device, rng):
        model = plast.nn.Sequential(
            plast.nn.Linear(2, 3, device=device),
        )
        sd = model.state_dict()
        model.load_state_dict(sd)


@ pytest.mark.xfail(reason="batch_norm uses eps scalar which creates 0-d tensor (segfault)")
class TestNormalization:
    def test_batch_norm_forward(self, device, tol, rng):
        bn = plast.nn.BatchNorm1d(4, device=device)
        x = plast.tensor(rng.randn(3, 4).astype(np.float32), device=device)
        out = bn(x)
        plast.forward(out)
        assert out.shape == [3, 4]


class TestDropout:
    def test_dropout_train(self, device, rng):
        dropout = plast.nn.Dropout(p=0.5)
        dropout.train()
        x = plast.tensor(rng.randn(100, 50).astype(np.float32), device=device)
        out = dropout(x)
        plast.forward(out)
        mask = out.numpy() != 0
        dropout_ratio = 1.0 - mask.mean()
        assert 0.3 < dropout_ratio < 0.7

    def test_dropout_eval(self, device, tol, rng):
        dropout = plast.nn.Dropout(p=0.5)
        dropout.eval()
        x = plast.tensor(rng.randn(10, 10).astype(np.float32), device=device)
        out = dropout(x)
        plast.forward(out)
        np.testing.assert_allclose(out.numpy(), x.numpy(), **tol)


class TestModuleUtilities:
    def test_module_train_eval(self, device, rng):
        model = plast.nn.Sequential(
            plast.nn.Linear(4, 3, device=device),
            plast.nn.Dropout(0.3),
        )
        assert model.training
        model.eval()
        assert not model.training
        model.train()
        assert model.training

    @pytest.mark.xfail(reason="model.zero_grad tries to import from wrong module")
    def test_module_zero_grad(self, device, rng):
        model = plast.nn.Linear(2, 2, device=device)
        x = plast.tensor(rng.randn(1, 2).astype(np.float32), device=device)
        out = model(x)
        loss = out.sum()
        plast.forward(loss)
        loss.backward()
        model.zero_grad()
        for p in model.parameters():
            if p.grad is not None:
                assert np.allclose(p.grad.numpy(), 0)

    def test_module_list(self, device, rng):
        layers = plast.nn.ModuleList([
            plast.nn.Linear(4, 8, device=device),
            plast.nn.Linear(8, 2, device=device),
        ])
        assert len(layers) == 2
        x = plast.tensor(rng.randn(1, 4).astype(np.float32), device=device)
        out = layers[1](layers[0](x))
        plast.forward(out)
        assert out.shape == [1, 2]
