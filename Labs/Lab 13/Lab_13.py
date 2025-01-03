import numpy as np
import matplotlib.pyplot as plt

# Perceptron Implementation
def perceptron(X, y, learning_rate=0.1, epochs=100):
    """Train a Perceptron model."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    for _ in range(epochs):
        for idx, x_i in enumerate(X):
            linear_output = np.dot(x_i, weights) + bias
            y_predicted = np.sign(linear_output)

            # Update rule
            update = learning_rate * (y[idx] - y_predicted)
            weights += update * x_i
            bias += update
    return weights, bias

def plot_perceptron_boundary(X, y, weights, bias):
    """Visualize the decision boundary for the Perceptron."""
    if weights[1] == 0:
        print("The second weight is zero; cannot plot a valid decision boundary.")
        return

    x1 = np.linspace(min(X[:, 0]) - 1, max(X[:, 0]) + 1, 100)
    x2 = -(weights[0] * x1 + bias) / weights[1]

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
    plt.plot(x1, x2, '-g')
    plt.title("Perceptron Decision Boundary")
    plt.show()


# Example dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([-1, 1, 1, -1])  # Binary labels

# Train Perceptron
weights, bias = perceptron(X, y)
plot_perceptron_boundary(X, y, weights, bias)


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Define the Neural Network for XOR
class XORNN(nn.Module):
    def __init__(self):
        super(XORNN, self).__init__()
        self.hidden = nn.Linear(2, 4)  # Hidden layer with 4 neurons
        self.output = nn.Linear(4, 1)  # Output layer
        self.activation = nn.Sigmoid()  # Sigmoid activation for simplicity

    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.activation(self.output(x))
        return x

# Function to train the XOR neural network
def train_xor_nn(X, y, epochs=1000, learning_rate=0.1):
    """Train a neural network to solve the XOR problem."""
    model = XORNN()  # Instantiate the model
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Optimizer

    # Convert input data to PyTorch tensors
    inputs = torch.Tensor(X)
    labels = torch.Tensor(y).reshape(-1, 1)

    # Training loop
    for epoch in range(epochs):
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss

        optimizer.zero_grad()  # Zero gradients
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        # Optionally, print loss for debugging
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")
    
    return model

# Function to visualize the XOR decision boundary
def visualize_xor_boundary(model, X, y):
    """Visualize the decision boundary for the XOR problem."""
    # Create a mesh grid of x1 and x2 values
    x1, x2 = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    grid = np.c_[x1.ravel(), x2.ravel()]  # Combine into input pairs

    # Predict outputs for the grid
    predictions = model(torch.Tensor(grid)).detach().numpy().reshape(x1.shape)

    # Plot the decision boundary
    plt.contourf(x1, x2, predictions, levels=50, cmap="RdYlBu", alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolor="k")
    plt.title("XOR Decision Boundary")
    plt.show()

# XOR Dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])  # XOR labels

# Train the neural network
model = train_xor_nn(X, y)

# Visualize the decision boundary
visualize_xor_boundary(model, X, y)
