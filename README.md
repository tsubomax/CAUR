# Class-Adaptive Uncertainty Revocation (CAUR) Framework

This repository provides the minimal Python implementation of the **Class-Adaptive Uncertainty Revocation (CAUR)** framework, a lightweight post-classification module for improving spatial consistency in remote sensing imagery.

## Features
- **Classifier Independent:** Can wrap any `scikit-learn` compatible classifier (e.g., Random Forest, SVM, kNN, etc.).
- **Class-Adaptive Thresholds:** Automatically optimizes entropy thresholds per class using Out-of-Bag (OOB) predictions (for tree-based models) or training set fallback.
- **Optuna Ready:** Structural hyperparameters (diffusion weight, spatial threshold, etc.) are exposed as standard arguments, allowing for seamless integration with Optuna.
- **Computationally Lightweight:** Improves spatial coherence without requiring expensive spatial-spectral model retraining.

## Installation

Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

A complete example demonstrating how to optimize CAUR parameters using Optuna is provided in `example_optuna.py`:

```bash
python example_optuna.py
```

### Basic Usage

```python
from sklearn.ensemble import RandomForestClassifier
from caur import CAURClassifier

# 1. Define your base estimator
base_model = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)

# 2. Wrap it with CAUR
model = CAURClassifier(
    estimator=base_model,
    diffusion_weight=0.3,
    spatial_threshold=3,
    reconstruction_iters=5
)

# 3. Fit the model 
# (Pass spatial_shape and train_idx for OOB threshold optimization)
# X_train, y_train are subsets of the full image features.
model.fit(X_train, y_train, spatial_shape=(H, W), train_idx=train_idx)

# 4. Predict
# X_full contains features for the entire H x W image
predictions = model.predict(X_full, spatial_shape=(H, W))
```

## Parameter Description

| Parameter | Default | Description |
|-----------|---------|-------------|
| `diffusion_weight` | `0.3` | Blend ratio for spatial probability diffusion. `0.0` disables diffusion. |
| `spatial_threshold`| `3` | Maximum number of same-class neighbors for a pixel to be considered isolated. |
| `reconstruction_iters`| `5` | Maximum number of iterative majority-vote reconstruction passes. |
| `percentiles_to_test`| `50..95` | Percentile range tested during class-adaptive entropy threshold optimization. |
