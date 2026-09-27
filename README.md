# MLBites

A lightweight Python machine learning library inspired by scikit-learn. This project contains educational and experimental implementations of common ML algorithms and dataset utilities.

## Overview

MLBites is a small learning project focused on building basic machine learning components from scratch using NumPy and familiar scikit-learn workflows. It includes implementations for:

- Linear regression
- Ridge regression
- K-nearest neighbors
- K-means clustering
- Decision tree structure (starter / experimental)
- Dataset loaders and synthetic data generation helpers

## Repository Structure

```text
mlbites/
├── cluster/
│   ├── __init__.py
│   ├── _base.py
│   └── kmeans.py
├── datasets/
│   ├── __init__.py
│   ├── loaders.py
│   ├── data/
│   │   ├── iris.csv
│   │   └── prepare_data.py
│   └── make_data/
│       └── make_classification.py
├── linear_model/
│   ├── _base.py
│   ├── linear_reg.py
│   └── ridge_reg.py
├── metrics/
│   ├── __init__.py
│   └── regression_loss/
│       ├── __init__.py
│       └── _base.py
├── neighbors/
│   ├── __init__.py
│   ├── _base.py
│   └── knn.py
├── tree/
│   ├── __init__.py
│   ├── _base.py
│   └── decision_tree.py
└── README.md