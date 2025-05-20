import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier 

# Step 1: Generate 100 random values in the range [0,1] 
np.random.seed(42)  # For reproducibility 
x = np.random.rand(100).reshape(-1, 1) 
print(x[:5]) 

# Step 2: Label the first 50 points 
labels = np.array([1 if xi <= 0.5 else 2 for xi in x[:50]]) 

# Step 3: Train KNN classifier 
k_values = [1, 2, 3, 4, 5, 20, 30] 
classified_labels = {} 
for k in k_values: 
    knn = KNeighborsClassifier(n_neighbors=k) 
    knn.fit(x[:50], labels)
    classified_labels[k] = knn.predict(x[50:])

# Step 4: Visualize the results 
plt.figure(figsize=(10, 6)) 
plt.scatter(x[:50], labels, color='blue', label='Training Data')
plt.scatter(x[50:], classified_labels[1], color='red', marker='x', label='Classified 
Data (k=1)') 
plt.xlabel('X values') 
plt.ylabel('Class') 
plt.title('KNN Classification of Random Values') 
plt.legend() 
plt.show() 

# Print classification results for different k values 
for k in k_values: 
    print(f"Classification results for k={k}: {classified_labels[k]}")