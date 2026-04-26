# Wine Classification using ML

Multiclass classification on the **WineQT** dataset using **from-scratch** machine learning models:
- Logistic Regression (One-vs-Rest)
- SVM (One-vs-Rest)
- Decision Tree (One-vs-Rest)

The project loads `dataset/WineQT.csv`, shuffles the data, performs an 80/20 train-test split, applies **min-max scaling** (fit on train only), trains each model, and prints train/test accuracy.

## Repository Structure

```text
.
├── main.py
├── dataset_loader.py
├── preprocessing.py
├── multiclass.py
├── metrics.py
├── models/
│   ├── __init__.py
│   ├── logistic_regression.py
│   ├── svm.py
│   └── decision_tree.py
└── dataset/
    └── WineQT.csv
```

## Requirements

- Python 3.8+ (recommended)
- `numpy`

Install numpy:

```bash
pip install numpy
```

## Run

From the repository root:

```bash
python main.py
```

### What you should see
`main.py` prints:
- Train/test class distribution
- Train accuracy + test accuracy for each model:
  - `LogReg-OvR`
  - `SVM-OvR`
  - `Tree-OvR`

## How it works (high level)

- **Dataset loading:** `dataset_loader.py`
  - `load_wintqt_csv("dataset/WineQT.csv")` loads features `X` and labels `y` (quality).
- **Preprocessing:** `preprocessing.py`
  - `min_max_scale_fit(X_train)` computes per-feature min/max on training data.
  - `min_max_scale_transform(...)` scales train and test using the training scaler.
- **Multiclass strategy:** `multiclass.py`
  - `OneVsRest` trains one binary classifier per class and picks the class with the best score.
- **Models:** `models/`
  - `LogisticRegression` exposes `predict_proba`
  - `SVM` exposes `decision_function`
  - `DecisionTree` uses `predict`

## Notes

- Ensure the dataset file exists at: `dataset/WineQT.csv`
- Hyperparameters are currently defined in `main.py`:
  - Logistic Regression: `lr=0.05, epochs=3000`
  - SVM: `lr=0.001, epochs=4000`
  - Decision Tree: `max_depth=15`
