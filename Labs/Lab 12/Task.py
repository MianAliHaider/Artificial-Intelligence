import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cross_entropy_loss(y_true, y_pred): 
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

def gradient_descent(X, y, weights, learning_rate, iterations): 
    m = len(y)
    for _ in range(iterations):
        predictions = sigmoid(np.dot(X, weights))
        gradient = np.dot(X.T, (predictions - y)) / m
        weights -= learning_rate * gradient
    return weights

def predict(X, weights):
    probabilities = sigmoid(np.dot(X, weights))
    return (probabilities >= 0.5).astype(int) 

def logistic_regression(X, y, learning_rate=0.1, iterations=1000): 
    weights = np.zeros(X.shape[1])
    weights = gradient_descent(X, y, weights, learning_rate, iterations)
    return weights

def evaluate(y_true, y_pred): 
    return np.mean(y_true == y_pred) 

X = np.array([[0.1, 1.1],
              [1.2, 0.9],
              [1.5, 1.6],
              [2.0, 1.8],
              [2.5, 2.1],
              [0.5, 1.5],
              [1.8, 2.3],
              [0.2, 0.7],
              [1.9, 1.4],
              [0.8, 0.6]])

y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 0])
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std
learning_rate = 0.1
iterations = 1000
weights = logistic_regression(X, y, learning_rate=learning_rate, iterations=iterations)
y_pred = predict(X, weights)
accuracy = evaluate(y, y_pred)
print(f"Trained Weights: {weights}")
print(f"Predictions: {y_pred}")
print(f"Accuracy: {accuracy:.2f}")


import matplotlib.pyplot as plt

for i in range(len(y)):
    if y[i] == 0:
        plt.scatter(X[i, 0], X[i, 1], color='red', label='Class 0' if i == 0 else "")
    else:
        plt.scatter(X[i, 0], X[i, 1], color='blue', label='Class 1' if i == 0 else "")

x1 = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
x2 = -(weights[0] * x1 + weights[1] * 0 + weights[-1]) / weights[1]  # Decision boundary equation
plt.plot(x1, x2, color='green', label='Decision Boundary')

plt.xlabel('X1 (Standardized)')
plt.ylabel('X2 (Standardized)')
plt.legend()
plt.title('Logistic Regression Decision Boundary')
plt.show()
