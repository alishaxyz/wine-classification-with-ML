import numpy as np
from collections import Counter

from models import LogisticRegression, SVM, DecisionTree
from multiclass import OneVsRest
from dataset_loader import load_wintqt_csv
from preprocessing import min_max_scale_fit, min_max_scale_transform

# Load dataset
X, y = load_wintqt_csv("dataset/WineQT.csv")

# Shuffle (numpy allowed)
rng = np.random.default_rng(42)
perm = rng.permutation(len(X))
X = [X[i] for i in perm]
y = [y[i] for i in perm]

# Train/test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("Train class distribution:", Counter(y_train))
print("Test class distribution :", Counter(y_test))

# Scale using train stats only
scaler = min_max_scale_fit(X_train)
X_train = min_max_scale_transform(X_train, scaler)
X_test = min_max_scale_transform(X_test, scaler)

models = [
    ("LogReg-OvR", OneVsRest(LogisticRegression, lr=0.05, epochs=3000)),
    ("SVM-OvR", OneVsRest(SVM, lr=0.001, epochs=4000)),
    ("Tree-OvR", OneVsRest(DecisionTree, max_depth=15)),
]

for name, model in models:
    model.fit(X_train, y_train)

    preds_test = model.predict(X_test)
    preds_train = model.predict(X_train)

    test_acc = sum(1 for i in range(len(y_test)) if preds_test[i] == y_test[i]) / len(y_test)
    train_acc = sum(1 for i in range(len(y_train)) if preds_train[i] == y_train[i]) / len(y_train)

    print(name)
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Test  accuracy: {test_acc:.4f}")
    print("-" * 40)
