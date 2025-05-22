#Pgm 7

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import warnings 
warnings.filterwarnings('ignore')

plt.figure(figsize=(10, 10))
X, y = fetch_california_housing(as_frame=True).data[["AveRooms"]], fetch_california_housing(as_frame=True).target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
m = LinearRegression().fit(X_train, y_train)
y_pred = m.predict(X_test)
plt.subplot(2,1,1)
plt.scatter(X_test, y_test, s=10)
plt.plot(X_test, y_pred, 'r')
plt.title("Linear Regression")
print("Linear MSE:", mean_squared_error(y_test, y_pred), "R2:", r2_score(y_test, y_pred))

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
                 sep='\s+', names=["mpg","cyl","disp","hp","wt","acc","yr","org"], na_values="?").dropna()
X, y = df[["hp"]], df["mpg"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
m = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(X_train, y_train)
xr = np.linspace(X.min(), X.max(), 300).reshape(-1,1)
plt.subplot(2,1,2)
plt.scatter(X, y, s=10, alpha=0.3)
plt.plot(xr, m.predict(xr), 'g')
plt.title("Polynomial Regression")
print("Poly MSE:", mean_squared_error(y_test, m.predict(X_test)), "R2:", r2_score(y_test, m.predict(X_test)))
plt.tight_layout()
plt.show()