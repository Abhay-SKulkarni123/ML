import numpy as np 
import matplotlib.pyplot as plt 
 
def gaussian_kernel(x, x_query, tau): 
    """Compute the Gaussian weight for each training sample.""" 
    return np.exp(-np.square(x - x_query) / (2 * tau**2)) 
 
def locally_weighted_regression(x_train, y_train, x_query, tau): 
    """Perform Locally Weighted Regression (LWR) for a given query point.""" 
    m = len(x_train) 
    W = np.diag(gaussian_kernel(x_train, x_query, tau)) 
     
    X_bias = np.c_[np.ones(m), x_train]
    theta = np.linalg.pinv(X_bias.T @ W @ X_bias) @ (X_bias.T @ W @ y_train) 
     
    return np.array([1, x_query]) @ theta
 
# Generate synthetic dataset 
np.random.seed(42) 
x_train = np.linspace(0, 10, 100) 
y_train = np.sin(x_train) + np.random.normal(0, 0.2, 100)  # Sinusoidal data 
with noise 
 
# Define tau (bandwidth parameter) 
tau_values = [0.1, 0.5, 1, 5] 
x_test = np.linspace(0, 10, 100)
 
plt.figure(figsize=(12, 8)) 
for tau in tau_values: 
    y_pred = np.array([locally_weighted_regression(x_train, y_train, xq, tau) 
for xq in x_test]) 
    plt.plot(x_test, y_pred, label=f'tau={tau}') 
 
# Plot training data 
plt.scatter(x_train, y_train, color='black', label='Training Data', alpha=0.5) 
plt.xlabel('X') 
plt.ylabel('Y') 
plt.title('Locally Weighted Regression (LWR) with Different Tau Values') 
plt.legend() 
plt.show()