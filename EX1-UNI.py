import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D

# Set random seed
np.random.seed(42)

# --- 1. UNIVARIATE REGRESSION ---
# Simulate data
X_uni = np.random.rand(100, 1) * 10
y_uni = 3 * X_uni.squeeze() + 7 + np.random.randn(100) * 2

# Fit model
model_uni = LinearRegression().fit(X_uni, y_uni)
y_uni_pred = model_uni.predict(X_uni)

# Plot
plt.figure(figsize=(6,4))
plt.scatter(X_uni, y_uni, label="Actual", color="blue")
plt.plot(X_uni, y_uni_pred, label="Predicted", color="red")
plt.title("Univariate Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# Metrics
print("Univariate Regression:")
print("MSE:", mean_squared_error(y_uni, y_uni_pred))
print("R² Score:", r2_score(y_uni, y_uni_pred))
print()


# --- 2. BIVARIATE REGRESSION ---
# Simulate data
X1 = np.random.rand(100, 1) * 10
X2 = np.random.rand(100, 1) * 5
X_bi = np.hstack([X1, X2])
y_bi = 2 * X1.squeeze() + 4 * X2.squeeze() + 5 + np.random.randn(100) * 2

# Fit model
model_bi = LinearRegression().fit(X_bi, y_bi)
y_bi_pred = model_bi.predict(X_bi)

# 3D plot
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X1, X2, y_bi, c='blue', label='Actual')
ax.scatter(X1, X2, y_bi_pred, c='red', label='Predicted', alpha=0.5)
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("y")
ax.set_title("Bivariate Regression")
plt.legend()
plt.show()

# Metrics
print("Bivariate Regression:")
print("MSE:", mean_squared_error(y_bi, y_bi_pred))
print("R² Score:", r2_score(y_bi, y_bi_pred))
print()


# --- 3. MULTIVARIATE REGRESSION ---
# Simulate data
X_multi = np.random.rand(100, 5)
coeffs = np.array([2, -1, 3, 0.5, 4])
y_multi = X_multi @ coeffs + 10 + np.random.randn(100) * 2

# Fit model
model_multi = LinearRegression().fit(X_multi, y_multi)
y_multi_pred = model_multi.predict(X_multi)

# Plot residuals
plt.figure(figsize=(6,4))
sns.histplot(y_multi - y_multi_pred, kde=True)
plt.title("Residuals - Multivariate Regression")
plt.xlabel("Residuals")
plt.show()

# Metrics
print("Multivariate Regression:")
print("MSE:", mean_squared_error(y_multi, y_multi_pred))
print("R² Score:", r2_score(y_multi, y_multi_pred))
print()
