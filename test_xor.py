import plast as p
import numpy as np

def train_xor():
    print("Testing XOR training in Python on CUDA...")
    p.init_arenas(device=p.Device.CUDA)

    # Data
    x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

    # Allocate X and Y in the persistent arena because they are reused across epochs
    X = p.tensor(x_data, device=p.Device.CUDA, persistent=True)
    Y = p.tensor(y_data, device=p.Device.CUDA, persistent=True)

    # Model
    hidden_size = 8
    model = p.nn.Sequential(
        p.nn.Linear(2, hidden_size, device=p.Device.CUDA),
        p.nn.ReLU(),
        p.nn.Linear(hidden_size, 1, device=p.Device.CUDA)
    )

    loss_fn = p.nn.MSELoss()
    optimizer = p.optim.SGD(model.parameters(), lr=0.05)

    # Training
    for epoch in range(10001):
        optimizer.zero_grad()

        # Forward
        preds = model(X)
        loss = loss_fn(preds, Y)

        # Execute graph forward
        loss.forward()

        if epoch % 1000 == 0:
            l = loss.item()
            print(f"Epoch {epoch:4d} | Loss: {l:.6f}")

        # Backward
        loss.backward()

        # Step
        optimizer.step()

        # Reset transient arenas to clear temporary activation and gradient buffers
        p.reset_transient_arenas()

    # Final result
    preds_test = model(X)
    preds = preds_test.numpy()
    print("\nFinal Predictions:")
    print(preds)

    expected = np.array([[0], [1], [1], [0]])
    if np.allclose(preds, expected, atol=0.2):
        print("\nSUCCESS: XOR converged on CUDA!")
    else:
        print("\nFAILURE: XOR did not converge correctly on CUDA.")

if __name__ == "__main__":
    train_xor()
