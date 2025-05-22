#Pgm 10

import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_scaled)
labels = np.where(kmeans.labels_ == 1, 0, 1)

print("Confusion Matrix:\n", confusion_matrix(y, labels))
print("\nAccuracy:", round(accuracy_score(y, labels), 2))

X_pca = PCA(n_components=2).fit_transform(X_scaled)
centers = PCA(n_components=2).fit(X_scaled).transform(kmeans.cluster_centers_)

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centroids')
plt.title("K-Means Clustering (Breast Cancer - PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)
plt.show()