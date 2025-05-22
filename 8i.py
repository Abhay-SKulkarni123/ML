#Pgm 8

import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
names = load_breast_cancer().feature_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(criterion="entropy", max_depth=7).fit(X_train, y_train)
print("Accuracy:", round(accuracy_score(y_test, model.predict(X_test)), 2))

plt.figure(figsize=(14, 8))
plot_tree(model, feature_names=names, class_names=["malignant", "benign"], filled=True)
plt.title("Decision Tree (Breast Cancer)"); plt.show()

print("\nRules:\n", export_text(model, feature_names=list(names)))

sample = np.array([[12.45,14.23,82.57,477.1,0.0823,0.0567,0.0379,0.0291,0.1556,0.0584,
0.2789,0.7225,1.75,24.58,0.00521,0.01145,0.0127,0.0091,0.0164,0.00247,
13.67,17.12,88.03,565.2,0.0981,0.1347,0.1342,0.0773,0.2871,0.0702]])
print("\nPredicted Class:", ["malignant", "benign"][model.predict(sample)[0]])