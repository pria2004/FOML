# Manufacturing Quality Control using PCA (Layman Friendly Code)

# Step 1: Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Step 2: Simulate Sensor Data
# 250 Good Products and 50 Faulty Products
np.random.seed(42)

# Good products have stable sensor values
good_products = np.random.normal(loc=0, scale=1, size=(250, 6))

# Faulty products have more variation (higher spread)
faulty_products = np.random.normal(loc=0, scale=3, size=(50, 6))

# Combine into one dataset
all_products = np.vstack((good_products, faulty_products))

# Create Labels: 0 = Good, 1 = Faulty
labels = np.array([0]*250 + [1]*50)

# Convert to DataFrame for readability
sensor_df = pd.DataFrame(all_products, columns=[f'Sensor_{i}' for i in range(1, 7)])
sensor_df['Label'] = labels

# Step 3: Standardize the Sensor Data (important for PCA)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(sensor_df.drop('Label', axis=1))

# Step 4: Apply PCA to reduce 6 sensor values into 2
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Print how much information we kept
print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)
print(f"Total Variance Captured by PC1 & PC2: {np.sum(pca.explained_variance_ratio_):.2f}")

# Step 5: Visualize Good vs Faulty Products in 2D using PCA
plt.figure(figsize=(8,6))
sns.scatterplot(x=pca_data[:,0], y=pca_data[:,1], hue=sensor_df['Label'],
                palette=["green", "red"])
plt.title("PCA - Good vs Faulty Products")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Product Type", labels=["Good", "Faulty"])
plt.grid(True)
plt.show()

# Step 6: Use KMeans to Automatically Group Products (No labels used)
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(pca_data)

# Visualize the Machine's Clustering
plt.figure(figsize=(8,6))
sns.scatterplot(x=pca_data[:,0], y=pca_data[:,1], hue=clusters, palette='coolwarm')
plt.title("PCA + KMeans Clustering (Auto-grouped)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Cluster")
plt.grid(True)
plt.show()

# Step 7: See which sensors influence the data the most
pca_loadings = pd.DataFrame(pca.components_,
                            columns=sensor_df.columns[:-1],
                            index=['PC1', 'PC2'])
print("\nSensor Contribution to Principal Components (PCA Loadings):")
print(pca_loadings)