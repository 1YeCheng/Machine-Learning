# Lab 1: Linear Models

This lab contains small-sample binary classification experiments on the Watermelon 3.0a dataset.

## Contents

- `3.3/`: Logistic regression experiments, including training-set resubstitution, leave-one-out cross-validation, bootstrap out-of-bag estimation, and quadratic feature expansion.
- `3.5/`: Linear discriminant analysis experiments with multiple validation settings.

## Run

```bash
cd 3.3
python src/logistic_regression_watermelon.py
python src/logistic_regression_quadratic_5fold.py

cd ../3.5
python src/linear_discriminant_analysis_watermelon.py
```
