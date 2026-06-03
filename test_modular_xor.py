import plast
import numpy as np


def run_xor():
    print("Testing modular XOR training with experiment tracking...")

    # 1. Initialize arenas (meta 10MB, data 100MB) on CPU
    plast.init_arenas(device=plast.Device.CPU)

    # 2. Setup config
    config = plast.experiment.ExperimentConfig(
        name="modular_xor",
        model={"hidden_size": 8, "activation": "ReLU"},
        training={"lr": 0.05, "epochs": 1000, "batch_size": 4, "optimizer": "SGD"},
        device="CPU",
    )

    # 3. Create experiment tracker
    tracker = plast.experiment.ExperimentTracker(config)

    # 4. Prepare data
    x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

    dataset = plast.data.TensorDataset(x_data, y_data)
    loader = plast.data.DataLoader(dataset, batch_size=4, shuffle=False)

    # 5. Define Model
    model = plast.nn.Sequential(plast.nn.Linear(2, 8), plast.nn.ReLU(), plast.nn.Linear(8, 1))

    loss_fn = plast.nn.MSELoss()
    optimizer = plast.optim.Adam(model.parameters(), lr=0.01)

    # 6. Training loop
    for epoch in range(1001):
        epoch_loss = 0.0

        for x_batch, y_batch in loader:
            optimizer.zero_grad()

            # Forward
            preds = model(x_batch)
            loss = loss_fn(preds, y_batch)

            # Execute computation graph forward
            plast.forward(loss)

            loss_val = loss.item()
            epoch_loss += loss_val

            # Backward
            loss.backward()

            # Step
            optimizer.step()

            # IMPORTANT: reset transient arenas to prevent memory growth!
            # Since all weights are Parameters (allocated in the persistent arena),
            # they are preserved perfectly. Transient arenas are cleared for the next step.
            plast.reset_transient_arenas()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss:.6f}")
            # Log epoch metrics to tracker
            tracker.log_epoch(epoch, {"train_loss": epoch_loss}, model=model)

    # 7. Final predictions
    x_test = plast.tensor(x_data)
    preds_test = model(x_test)
    plast.forward(preds_test)
    preds_np = preds_test.numpy()

    print("\nFinal predictions:")
    print(preds_np)

    tracker.finish()

    # Simple validation
    expected = np.array([[0], [1], [1], [0]])
    if np.allclose(preds_np, expected, atol=0.2):
        print("\nSUCCESS: XOR converged successfully in modular format!")
    else:
        print("\nFAILURE: XOR did not converge.")


if __name__ == "__main__":
    run_xor()
