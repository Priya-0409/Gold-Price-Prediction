# Step 1: Import Libraries
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Create Synthetic Data (Simple Linear Relationship)
X = torch.linspace(-5, 5, 100).reshape(-1, 1)  # Input (100 samples)
y = 2 * X + 1 + torch.randn(X.shape) * 0.5     # Output (with noise)

# Plot the data
plt.scatter(X.numpy(), y.numpy(), label="Data")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# Step 3: Define a Simple Linear Model
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # 1 input, 1 output

    def forward(self, x):
        return self.linear(x)

model = LinearModel()

# Step 4: Train the Model
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent


losses = []
epochs = 100

for epoch in range(epochs):
    # Forward pass
    predictions = model(X)
    loss = criterion(predictions, y)
    
    # Backward pass
    optimizer.zero_grad()  # Reset gradients
    loss.backward()        # Compute gradients
    optimizer.step()       # Update weights
    
    losses.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Plot training loss
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

# Step 5: Evaluate the Model
model.eval()  # Set model to evaluation mode
with torch.no_grad():
    predicted_y = model(X).numpy()

# Plot predictions vs true data
plt.scatter(X.numpy(), y.numpy(), label="True Data")
plt.plot(X.numpy(), predicted_y, color='red', label="Predicted Line")
plt.legend()
plt.show()

# Step 6: Save the Model
torch.save(model.state_dict(), "linear_model.pth")
print("Model saved as 'linear_model.pth'")
