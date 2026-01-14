import plast
import numpy as np

def train_xor():
    print("Testing XOR training in Python on CUDA...")
    meta, data = plast.init_arenas(device=plast.Device.CUDA)
    
    # Data
    x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
    y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)
    
    X = plast.tensor(x_data, device=plast.Device.CUDA)
    Y = plast.tensor(y_data, device=plast.Device.CUDA)
    
    # Model
    from plast import nn, optim
    hidden_size = 8
    model = nn.Module()
    model.l1 = nn.Linear(2, hidden_size, device=plast.Device.CUDA)
    model.relu = nn.ReLU()
    model.l2 = nn.Linear(hidden_size, 1, device=plast.Device.CUDA)
    
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Training
    for epoch in range(10001):
        optimizer.zero_grad()
        
        # Forward
        h1 = model.relu(model.l1(X))
        logits = model.l2(h1)
        loss = loss_fn(logits, Y)
        
        # Execute graph
        from plast.plast_core import forward, backward, set_ones_grad
        forward(loss)
        
        if epoch % 1000 == 0:
            l = loss.numpy()[0]
            print(f"Epoch {epoch}, Loss: {l:.6f}")
            
        # Backward
        set_ones_grad(loss)
        backward(loss)
        
        # Step
        optimizer.step()

    # Final result
    h1 = model.relu(model.l1(X))
    logits = model.l2(h1)
    forward(logits)
    preds = logits.numpy()
    print("\nFinal Predictions:")
    print(preds)
    
    expected = np.array([[0],[1],[1],[0]])
    if np.allclose(preds, expected, atol=0.1):
        print("\nSUCCESS: XOR converged!")
    else:
        print("\nFAILURE: XOR did not converge correctly.")

if __name__ == "__main__":
    train_xor()
