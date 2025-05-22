#Pgm 9

import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

X, y = fetch_olivetti_faces(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GaussianNB().fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 2))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

fig, axes = plt.subplots(5, 8, figsize=(14, 8))
shown = set()
i = 0
for ax in axes.flat:
    while y[i] in shown:
        i += 1
    ax.imshow(X[i].reshape(64, 64), cmap='gray')
    ax.axis('off')
    ax.set_title(f'ID {y[i]}')
    shown.add(y[i])
    i += 1
plt.tight_layout()
plt.show()