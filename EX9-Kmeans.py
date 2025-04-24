import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ---------------------------
# K-MEANS CUSTOMER SEGMENTATION
# ---------------------------
customer_data = pd.DataFrame({
    'CustomerID': range(1, 11),
    'Annual Income (k$)': [15, 16, 17, 18, 90, 95, 88, 85, 60, 62],
    'Spending Score (1-100)': [39, 81, 6, 77, 40, 90, 76, 55, 50, 48]
})

X = customer_data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Elbow Method
wcss = []
for i in range(1, 6):
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(X)
    wcss.append(km.inertia_)

plt.plot(range(1, 6), wcss, marker='o')
plt.title('Elbow Method - Optimal K')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fit KMeans
kmeans = KMeans(n_clusters=2, random_state=0)
customer_data['Segment'] = kmeans.fit_predict(X)

# Cluster Visualization
plt.figure(figsize=(8, 5))
sns.scatterplot(data=customer_data, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Segment', palette='Set2', s=100)
plt.title('Customer Segmentation')
plt.grid(True)
plt.show()

print("\nCustomer Cluster Summary:\n", customer_data.groupby('Segment').mean(numeric_only=True))

# ---------------------------
# KNN: PRODUCT RECOMMENDATION
# ---------------------------
data = pd.DataFrame({
    'Age': [25, 30, 45, 35, 52, 23, 40, 60, 22, 48],
    'Income': [40, 50, 80, 60, 90, 35, 70, 100, 38, 85],
    'Bought': [0, 0, 1, 0, 1, 0, 1, 1, 0, 1]
})

X = data[['Age', 'Income']]
y = data['Bought']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
print("\nKNN Accuracy:", acc)

cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", cr)

# Confusion matrix heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('KNN Confusion Matrix')
plt.show()

# Predict for a new customer
new_customer = np.array([[34, 75]])  # Age = 34, Income = 75
prediction = knn.predict(new_customer)
print("Prediction for new customer (Age=34, Income=75):", "Will Buy" if prediction[0] == 1 else "Will Not Buy")