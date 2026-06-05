import plast
import numpy as np
import gzip
import struct
import urllib.request
import os.path

HAS_CUDA = hasattr(plast.Device, "CUDA")
DEVICE = plast.Device.CUDA if HAS_CUDA else plast.Device.CPU

DATA_DIR = "/tmp/plast_mnist"
MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"


def _download(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        url = MNIST_URL + filename
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, path)
    return path


def _parse_idx(path):
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic == 2051:
            rows, cols = struct.unpack(">II", f.read(8))
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
        else:
            data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def load_mnist():
    os.makedirs(DATA_DIR, exist_ok=True)

    train_imgs = _parse_idx(_download("train-images-idx3-ubyte.gz")).astype(np.float32) / 255.0
    train_labels = _parse_idx(_download("train-labels-idx1-ubyte.gz"))
    test_imgs = _parse_idx(_download("t10k-images-idx3-ubyte.gz")).astype(np.float32) / 255.0
    test_labels = _parse_idx(_download("t10k-labels-idx1-ubyte.gz"))

    return train_imgs, train_labels, test_imgs, test_labels


def accuracy(logits_np, labels_np):
    preds = np.argmax(logits_np, axis=1)
    return np.mean(preds == labels_np)


def train():
    x_train, y_train, x_test, y_test = load_mnist()
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    # One-hot encode targets (MSE expects float targets)
    y_train_onehot = np.zeros((len(y_train), 10), dtype=np.float32)
    y_train_onehot[np.arange(len(y_train)), y_train] = 1.0

    print(f"Train: {len(x_train)} samples, Test: {len(x_test)} samples")

    plast.init_arenas(meta_size_mb=50, data_size_mb=500, device=DEVICE)

    model = plast.nn.Sequential(
        plast.nn.Linear(784, 128, device=DEVICE),
        plast.nn.ReLU(),
        plast.nn.Linear(128, 10, device=DEVICE),
    )
    loss_fn = plast.nn.MSELoss()
    optimizer = plast.optim.SGD(model.parameters(), lr=0.1)

    batch_size = 64
    n_batches = len(x_train) // batch_size

    @plast.jit
    def train_step(batch_idx):
        X = plast.tensor(x_train[batch_idx], device=DEVICE)
        Y = plast.tensor(y_train_onehot[batch_idx], device=DEVICE)

        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, Y)
        plast.forward(loss)
        loss.backward()
        optimizer.step()
        return loss

    for epoch in range(5):
        indices = np.random.permutation(len(x_train))
        epoch_loss = 0.0

        for i in range(n_batches):
            batch_idx = indices[i * batch_size : (i + 1) * batch_size]

            loss = train_step(batch_idx)
            plast.reset_transient_arenas()

            epoch_loss += loss.item()

        print(f"Epoch {epoch}, Loss: {epoch_loss / n_batches:.6f}")

    print("\nEvaluating on test set...")
    correct = 0
    for i in range(0, len(x_test), batch_size):
        X = plast.tensor(x_test[i : i + batch_size], device=DEVICE)
        pred = model(X)
        plast.forward(pred)
        correct += np.sum(np.argmax(pred.numpy(), axis=1) == y_test[i : i + batch_size])

    acc = correct / len(x_test)
    print(f"Test Accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    train()
