# CAUR Framework for Hyperspectral Image Classification

Code repository accompanying the paper submitted to **Applied Sciences** (MDPI).

## Overview

This repository contains Python code for the **Class-wise Accuracy Update and Refinement (CAUR)** framework. The CAUR framework is an iterative multiclass classification technique designed to systematically improve classification accuracy on hyperspectral satellite imagery by reclassifying low-confidence predictions using a cascade mechanism. The methods are evaluated against standard machine learning classifiers on mineral mapping tasks at the Cuprite mining district, Nevada, USA, and land cover classification at Mito city, Ibaraki, Japan.

---

## Directory Structure

```
Github/
├── README.md                      # This file
├── advanced_cascade_acc11.py      # Core CAUR Framework (CascadedACC11 class)
├── custom_models.py               # Custom wrappers for baseline classifiers (1D-CNN, TabNet, XGBoost, etc.)
├── pipeline_utils.py              # Utilities for data loading, model training, and accuracy evaluation
├── pipeline_config.py             # Configuration and hyperparameter settings
└── run_validation.py              # Main execution script for validation
```

---

## Script Descriptions

### `advanced_cascade_acc11.py` — Core CAUR Framework
Contains the `CascadedACC11` class, which implements the CAUR framework. The framework iteratively assigns classes based on their cross-validation F1-scores, cascades unclassified samples to subsequent models, and features an adaptive threshold mechanism to fallback on one-vs-rest probabilities.

### `run_validation.py` — Validation Pipeline
The main execution script to run the CAUR framework and baseline models on dataset features and labels. It evaluates performance across various datasets, comparing the baseline models to their CAUR-enhanced counterparts. Outputs classification metrics (Overall Accuracy, Macro F1, Kappa, MCC).

### `pipeline_utils.py` — Pipeline Utilities
Contains shared logic for reading classification data, training models across K-Fold cross-validation, applying SMOTE for class imbalance, and computing evaluation metrics.

### `custom_models.py` — Baseline Classifiers
Defines wrappers for various baseline algorithms so they seamlessly integrate into the pipeline. Includes PyTorch-based 1D-CNN, TabNet, XGBoost, among standard scikit-learn models.

### `pipeline_config.py` — Configuration Settings
Stores global variables, hyperparameters for all classifiers, and the definitions of the validation datasets.

---

## Requirements

```
Python >= 3.8
numpy
scikit-learn
xgboost
lightgbm
pytorch (torch)
pytorch-tabnet
imbalanced-learn
```

Install dependencies:
```bash
pip install numpy scikit-learn xgboost lightgbm torch pytorch-tabnet imbalanced-learn
```

---

## How to Run

### 1. Configure Input Data Paths
Before running, edit the dataset configurations inside `pipeline_config.py` to point to your local feature data and labels. Ensure your datasets are formatted properly for ingestion by `pipeline_utils.py`.

### 2. Run Validation Script
```bash
python run_validation.py
```
This will train the models, apply the CAUR framework, and output the validation results to the `results/` directory as CSV files.

---

## Data Availability

AVIRIS data are publicly available from the NASA AVIRIS Data Portal (https://aviris.jpl.nasa.gov/dataportal/). ASTER and EMIT data are available from the NASA EARTHDATA Portal (https://search.earthdata.nasa.gov/). HISUI data can be obtained through the Tellus platform (https://www.tellusxdp.com/).

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
