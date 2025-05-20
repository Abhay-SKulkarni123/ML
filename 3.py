import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris 
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler 

# Implement PCA for the Iris dataset 
print("Implementing PCA for the Iris dataset...") 
iris = load_iris() 
iris_data = iris.data 
iris_target = iris.target 
iris_feature_names = iris.feature_names 

# Standardize the data 
scaler = StandardScaler() 
iris_data_scaled = scaler.fit_transform(iris_data) 

# Apply PCA to reduce to 2 dimensions 
pca = PCA(n_components=2) 
iris_pca = pca.fit_transform(iris_data_scaled) 

# Create a DataFrame with the PCA results 
iris_pca_df = pd.DataFrame(data=iris_pca, columns=["Principal Component 1", "Principal Component 2"]) 
iris_pca_df['Target'] = iris_target 

# Visualize the PCA results 
plt.figure(figsize=(8, 6)) 
sns.scatterplot( x="Principal Component 1", y="Principal Component 2", hue="Target", data=iris_pca_df, palette="viridis", s=100, alpha=0.8) 
plt.title("PCA of Iris Dataset") 
plt.xlabel("Principal Component 1") 
plt.ylabel("Principal Component 2") 
plt.legend(title="Target", labels=iris.target_names) 
plt.grid(alpha=0.5) 
plt.show()