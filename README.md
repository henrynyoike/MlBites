# MLBites

A lightweight, educational Python machine learning library inspired by scikit-learn.  
MLBites implements core ML algorithms **from scratch** (primarily with NumPy) so you can study the math, experiment, and understand what happens under the hood.

> **Status**: Actively developed. Some modules are ready for learning and experimentation; others are scaffolding that will be completed soon.

---

## Features

| Module | Algorithms / Utilities | Status |
|--------|------------------------|--------|
| `linear_model` | `LinearRegression`, `RidgeRegression`, `LogisticRegression` | Implemented |
| `neighbors` | `KNeighborsClassifier` | Implemented |
| `cluster` | `KMeans` | Implemented |
| `tree` | `DecisionTreeClassifier`, `DecisionTreeRegressor` | Partial / experimental |
| `naive` | `GaussianNB` | Scaffolding (coming soon) |
| `metrics` | Regression metrics (MSE, MAE, R², RMSE, …) | Implemented |
| `datasets` | `load_iris`, `make_classification`, `make_regression` | Implemented |

---

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd mlbites

# Install dependencies
pip install numpy pandas scikit-learn
```

There is currently no PyPI package. Import the library by placing the project root on your `PYTHONPATH` or by running code from inside the project directory:

```python
import sys
sys.path.append("/path/to/mlbites")   # parent of the mlbites package
```

---

## Quick Start

```python
import numpy as np
from mlbites.linear_model.linear_reg import LinearRegression
from mlbites.metrics import mean_squared_error, r2_score
from mlbites.datasets.make.make_regression import make_regression

# Generate synthetic data
X, y = make_regression(n_samples=200, n_features=3)

# Train
model = LinearRegression()
model.fit(X, y, learning_rate=0.01, iterations=500)

# Predict & evaluate
y_pred = model.predict(X)
print("MSE :", mean_squared_error(y, y_pred))
print("R²  :", r2_score(y, y_pred))
```

---

## Package Structure

```text
mlbites/
├── linear_model/
│   ├── linear_reg.py          # LinearRegression
│   ├── ridge_reg.py           # RidgeRegression
│   └── regression.py          # LinearRegression + LogisticRegression
├── neighbors/
│   └── knn.py                 # KNeighborsClassifier
├── cluster/
│   └── kmeans.py              # KMeans
├── tree/
│   ├── decision_tree.py       # DecisionTreeClassifier (starter)
│   └── decision_tree_reg.py   # DecisionTreeRegressor
├── naive/
│   └── gaussian_naive.py      # GaussianNB (scaffolding)
├── metrics/
│   └── regression_loss/
│       └── _base.py           # MSE, MAE, R², RMSE, …
├── datasets/
│   ├── loaders.py             # load_iris
│   ├── make/
│   │   ├── make_classification.py
│   │   └── make_regression.py
│   └── data/
│       └── iris.csv
└── README.md
```

---

## Detailed Usage & Examples

### 1. Linear Regression

Ordinary least-squares style linear regression with optional gradient-descent refinement.

```python
from mlbites.linear_model.linear_reg import LinearRegression
import numpy as np

X = np.random.randn(100, 2)
y = (3 * X[:, 0] + 2 * X[:, 1] + 1).reshape(-1, 1)   # shape (n, 1)

model = LinearRegression()
coef, bias = model.fit(X, y, learning_rate=0.01, iterations=300)

print("Coefficients:", coef)
print("Bias       :", bias)

y_hat = model.predict(X)
```

**Notes**
- `y` must have shape `(n_samples, 1)`.
- The implementation first computes a closed-form solution, then optionally refines it with gradient descent.

### 2. Ridge Regression

Linear regression with gradient updates controlled by `alpha`.

```python
from mlbites.linear_model.ridge_reg import RidgeRegression
import numpy as np

X = np.random.uniform(size=(500, 4))
y = np.random.uniform(size=(500, 1))

model = RidgeRegression()
coef, bias = model.fit(X, y, alpha=0.01, iterations=200)
y_pred = model.predict(X)
```

### 3. Logistic Regression

Binary logistic regression with sigmoid activation.

```python
from mlbites.linear_model.regression import LogisticRegression
import numpy as np

X = np.random.randn(300, 3)
y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)

model = LogisticRegression()
model.fit(X, y, iterations=500, learning_rate=0.05)

proba = model.predict_proba(X)   # probabilities
labels = model.predict(X)        # 0/1 predictions
```

### 4. K-Nearest Neighbors Classifier

Minkowski-distance based KNN (default Euclidean, `p=2`).

```python
from mlbites.neighbors.knn import KNeighborsClassifier
from mlbites.datasets.make.make_classification import make_classification
import numpy as np

X, y = make_classification(n_samples=200, n_features=5, n_classes=3)
y = y.ravel()                    # 1-D labels for Counter

model = KNeighborsClassifier(n_neighbors=5, p=2)
model.fit(X, y)

preds = model.predict(X[:20])    # predict on a small subset
print(preds)
```

### 5. K-Means Clustering

```python
from mlbites.cluster.kmeans import KMeans
from sklearn.datasets import make_blobs
import numpy as np

X, _ = make_blobs(n_samples=300, n_features=2, centers=3)

model = KMeans(n_clusters=3, max_iters=100)
centroids, clusters = model.fit(X)

labels = model.predict(X)
print("Cluster labels:", labels[:10])
```

### 6. Decision Tree Regressor

Variance-based splitting (CART-style for regression).

```python
from mlbites.tree.decision_tree_reg import DecisionTreeRegressor
from mlbites.metrics import mean_squared_error
import numpy as np

np.random.seed(42)
X = np.random.randn(150, 3)
y = X[:, 0]**2 + 0.5 * X[:, 1] + np.random.randn(150) * 0.1

model = DecisionTreeRegressor()
model.fit(X, y)

y_pred = model.predict(X)
print("Train MSE:", mean_squared_error(y, y_pred))
```

> `DecisionTreeClassifier` currently provides the basic `Node` structure and a stub `fit`. Full split logic (Gini / entropy) is planned and will be finished soon.

### 7. Gaussian Naive Bayes (coming soon)

```python
from mlbites.naive.gaussian_naive import GaussianNB

# API preview — implementation in progress
model = GaussianNB()
# model.fit(X, y)
# preds = model.predict(X)
```

The intended design:
1. Group features by class.
2. Compute per-class mean and variance.
3. Predict via Gaussian class-conditional densities + prior.

### 8. Regression Metrics

```python
from mlbites.metrics import (
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    median_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_log_error,
    adjusted_r2_score,
)
import numpy as np

y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.1, 7.8])

print("MSE   :", mean_squared_error(y_true, y_pred))
print("MAE   :", mean_absolute_error(y_true, y_pred))
print("RMSE  :", root_mean_squared_error(y_true, y_pred))
print("R²    :", r2_score(y_true, y_pred))
print("Adj R²:", adjusted_r2_score(y_true, y_pred, features=1))
print("MedAE :", median_absolute_error(y_true, y_pred))
print("MAPE  :", mean_absolute_percentage_error(y_true, y_pred))
print("MSLE  :", mean_squared_log_error(np.abs(y_true), np.abs(y_pred)))
```

### 9. Datasets

#### Load Iris

```python
from mlbites.datasets.loaders import load_iris

iris = load_iris()
X, y = iris.data, iris.target
print(iris.feature_names)
print(iris.target_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# ['setosa', 'versicolor', 'virginica']
```

#### Synthetic Classification Data

```python
from mlbites.datasets.make.make_classification import make_classification

X, y = make_classification(
    n_samples=500,
    n_features=10,
    n_classes=3,
    n_repeated=0,
    shuffle=True,
)
print(X.shape, y.shape)   # (500, 10)  (500, 1)
```

#### Synthetic Regression Data

```python
from mlbites.datasets.make.make_regression import make_regression

X, y = make_regression(n_samples=200, n_features=5)
print(X.shape, y.shape)
```

---

## Design Philosophy

- **Educational first** – algorithms are written in plain NumPy so every matrix multiplication and gradient step is visible.
- **Familiar API** – most estimators expose the classic `fit` / `predict` interface used by scikit-learn.
- **Minimal dependencies** – NumPy is the core; pandas and scikit-learn appear only for convenience (data loading, comparison, synthetic helpers).
- **Incremental growth** – unfinished modules (`GaussianNB`, full Decision Tree Classifier, shared base classes, etc.) are left as clear stubs so future development stays straightforward.

---

## Requirements

- Python ≥ 3.9
- NumPy
- pandas (for the Iris loader)
- scikit-learn (optional – used in some examples and comparisons)

---

## Roadmap / Upcoming Work

- Complete `GaussianNB` (class-conditional means & variances, posterior prediction)
- Finish `DecisionTreeClassifier` (Gini / entropy splits, pruning)
- Shared base estimator classes (`_base.py` files)
- Classification metrics (accuracy, precision, recall, F1, confusion matrix)
- Proper package installation (`pyproject.toml` / `setup.cfg`)
- Unit tests for every estimator
- Documentation site / notebook tutorials

---

## Contributing

This is a learning project. Pull requests that:

- improve numerical stability,
- add missing algorithms,
- clean up the API, or
- expand the test suite

are very welcome.

---

## License

MIT

---

## Acknowledgments

Inspired by the excellent design of [scikit-learn](https://scikit-learn.org).  
Built for curiosity, experimentation, and understanding the fundamentals of machine learning.
