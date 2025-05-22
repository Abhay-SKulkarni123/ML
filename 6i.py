#Pgm 6

import numpy as np, matplotlib.pyplot as plt
def lwr(x_train, y_train, x_query, tau):
    weights = np.exp(-((x_train - x_query)**2) / (2 * tau**2))
    X = np.c_[np.ones_like(x_train), x_train]
    W = np.diag(weights)
    theta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ y_train)
    return np.dot([1, x_query], theta)

x = np.linspace(0, 10, 100)
x_test = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.2, 100)
taus = [0.1, 0.5, 1, 5]

plt.figure(figsize=(10, 6))
for tau in taus:
    y_pred = [lwr(x, y, xi, tau) for xi in x_test]
    plt.plot(x_test, y_pred, label=f'tau={tau}')
plt.scatter(x, y, c='black', s=15, label='Training Data', alpha=0.5)
plt.title("Locally Weighted Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()