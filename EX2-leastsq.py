import numpy as np
import matplotlib.pyplot as plt

# 1. Simulate data (y = 2x + 5 + noise)
np.random.seed(0)
X = np.random.rand(100) * 10
noise = np.random.randn(100)
y = 2 * X + 5 + noise

# 2. Least Squares Calculation
x_mean = np.mean(X)
y_mean = np.mean(y)

numerator = np.sum((X - x_mean) * (y - y_mean))
denominator = np.sum((X - x_mean) ** 2)
slope = numerator / denominator
intercept = y_mean - slope * x_mean

# 3. Predictions
y_pred = slope * X + intercept

# 4. Plot
plt.figure(figsize=(6,4))
plt.scatter(X, y, label="Actual", color="blue")
plt.plot(X, y_pred, color="red", label="Prediction (Line of Best Fit)")
plt.title("Simple Linear Regression - Least Squares")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# 5. Performance Metrics
mse = np.mean((y - y_pred) ** 2)
r2 = 1 - (np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2))

# 6. Output
print(f"Intercept: {intercept:.2f}")
print(f"Slope: {slope:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")
