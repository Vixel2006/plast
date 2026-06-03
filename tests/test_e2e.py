import numpy as np
import pytest
import plast


class TestXOR:
    def _make_xor_model(self, device):
        model = plast.nn.Sequential(
            plast.nn.Linear(2, 8, device=device),
            plast.nn.ReLU(),
            plast.nn.Linear(8, 1, device=device),
        )
        return model

    def test_xor_cpu(self):
        plast.reset_transient_arenas()
        plast.init_arenas(device=plast.Device.CPU)

        x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

        model = self._make_xor_model(plast.Device.CPU)
        loss_fn = plast.nn.MSELoss()
        optim = plast.optim.SGD(model.parameters(), lr=0.1)

        for epoch in range(5001):
            optim.zero_grad()
            X = plast.tensor(x_data, device=plast.Device.CPU)
            Y = plast.tensor(y_data, device=plast.Device.CPU)
            pred = model(X)
            loss = loss_fn(pred, Y)
            plast.forward(loss)
            loss.backward()
            optim.step()
            plast.reset_transient_arenas()
            if epoch % 1000 == 0:
                loss_val = loss.item()

        X = plast.tensor(x_data, device=plast.Device.CPU)
        pred = model(X)
        plast.forward(pred)
        preds = pred.numpy()
        expected = np.array([[0], [1], [1], [0]])
        assert np.allclose(preds, expected, atol=0.2), f"XOR failed on CPU: {preds}"

    @pytest.mark.cuda
    def test_xor_cuda(self):
        plast.reset_transient_arenas()
        plast.init_arenas(device=plast.Device.CUDA)

        x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

        model = self._make_xor_model(plast.Device.CUDA)
        loss_fn = plast.nn.MSELoss()
        optim = plast.optim.SGD(model.parameters(), lr=0.1)

        for epoch in range(5001):
            optim.zero_grad()
            X = plast.tensor(x_data, device=plast.Device.CUDA)
            Y = plast.tensor(y_data, device=plast.Device.CUDA)
            pred = model(X)
            loss = loss_fn(pred, Y)
            plast.forward(loss)
            loss.backward()
            optim.step()
            plast.reset_transient_arenas()
            if epoch % 1000 == 0:
                loss_val = loss.item()

        X = plast.tensor(x_data, device=plast.Device.CUDA)
        pred = model(X)
        plast.forward(pred)
        preds = pred.numpy()
        expected = np.array([[0], [1], [1], [0]])
        assert np.allclose(preds, expected, atol=0.2), f"XOR failed on CUDA: {preds}"


class TestSmallClassifier:
    def test_linear_classifier_cpu(self):
        plast.reset_transient_arenas()
        plast.init_arenas(device=plast.Device.CPU)

        np.random.seed(42)
        n = 100
        x_data = np.random.randn(n, 3).astype(np.float32)
        w_true = np.array([[2.0], [-1.0], [0.5]], dtype=np.float32)
        y_data = x_data @ w_true + 0.1 * np.random.randn(n, 1).astype(np.float32)

        model = plast.nn.Linear(3, 1, device=plast.Device.CPU)
        loss_fn = plast.nn.MSELoss()
        optim = plast.optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(1000):
            optim.zero_grad()
            X = plast.tensor(x_data, device=plast.Device.CPU)
            Y = plast.tensor(y_data, device=plast.Device.CPU)
            pred = model(X)
            loss = loss_fn(pred, Y)
            plast.forward(loss)
            loss.backward()
            optim.step()
            plast.reset_transient_arenas()

        X = plast.tensor(x_data, device=plast.Device.CPU)
        pred = model(X)
        plast.forward(pred)
        preds = pred.numpy()
        mse = np.mean((preds - y_data) ** 2)
        assert mse < 0.5, f"Classifier too inaccurate: MSE={mse}"

    @pytest.mark.slow
    def test_training_loss_monotonic_decrease(self):
        plast.reset_transient_arenas()
        plast.init_arenas(device=plast.Device.CPU)

        x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

        model = plast.nn.Sequential(
            plast.nn.Linear(2, 8),
            plast.nn.ReLU(),
            plast.nn.Linear(8, 1),
        )
        optim = plast.optim.SGD(model.parameters(), lr=0.1)

        prev_loss = float("inf")
        for epoch in range(500):
            optim.zero_grad()
            X = plast.tensor(x_data)
            Y = plast.tensor(y_data)
            pred = model(X)
            loss = plast.nn.MSELoss()(pred, Y)
            plast.forward(loss)
            loss.backward()
            optim.step()
            plast.reset_transient_arenas()
            current_loss = loss.item()
            if epoch > 10:
                assert current_loss <= prev_loss + 0.01, (
                    f"Loss increased at epoch {epoch}: {prev_loss} -> {current_loss}"
                )
            prev_loss = current_loss
