# Machine Learning Coursework Portfolio

![NumPy](https://img.shields.io/badge/NumPy-numerical%20computing-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-model%20evaluation-F7931E?logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-visualization-11557C)
![LIBSVM](https://img.shields.io/badge/LIBSVM-3.31-00599C)

This repository contains two undergraduate machine-learning coursework projects on classical supervised learning. The work covers from-scratch algorithm implementation, reproducible experimental workflows, model evaluation, and result visualisation, with complete records from data processing and model training to validation design and result analysis.

## Project Overview

| Project | Research Focus | Methods | Main Outputs |
|---|---|---|---|
| [`lab1`](lab1/README.md) | Small-sample binary classification on Watermelon 3.0a | From-scratch logistic regression, quadratic feature expansion, from-scratch LDA, LOOCV, stratified 5-fold CV, bootstrap OOB | Prediction tables, metric summaries, confusion matrices, PR curves |
| [`lab2`](lab2/README.md) | SVM/SVR experiments and multi-model classification comparison | LIBSVM, BP neural network, handwritten C4.5 decision tree, UCI Wine/WDBC classification | Model files, prediction outputs, metric tables, decision boundaries, confusion matrices, regression diagnostic plots |

## Methods and Experimental Design

The two projects cover complementary levels of machine-learning practice, from implementing model mechanisms to organising comparative experiments.

`lab1` focuses on the implementation and evaluation of classical linear classifiers. I implemented binary logistic regression from scratch, including Newton's method, numerically stable sigmoid/softplus computation, Gaussian elimination with partial pivoting, and optional L2 regularisation. I also implemented linear discriminant analysis, including class-mean estimation, shared covariance estimation, class priors, and covariance regularisation. To examine evaluation behaviour under a small-sample setting, the experiments use training-set resubstitution, leave-one-out cross-validation, stratified 5-fold cross-validation, and bootstrap out-of-bag estimation.

`lab2` focuses on model comparison and reproducible experimentation. I used LIBSVM to conduct linear-kernel and RBF-kernel SVM classification, as well as SVR regression. I also compared Linear SVM, RBF SVM, BP neural network, and a handwritten C4.5 decision tree on the UCI Wine and WDBC breast cancer diagnosis datasets. The Wine task is formulated as a three-class wine classification problem using 13 physicochemical features. The WDBC task is formulated as benign-versus-malignant tumour classification using 30 cytological diagnostic features. Apart from LIBSVM and the BP neural network, the C4.5 decision tree, experiment orchestration, metric computation, and plotting routines were implemented by myself.

## Selected Results

| Experiment | Dataset | Result |
|---|---|---|
| Quadratic-feature logistic regression | Watermelon 3.0a | 5-fold CV accuracy 0.7059, F1 0.7059 |
| Linear discriminant analysis | Watermelon 3.0a | 5-fold CV accuracy 0.6471, F1 0.6250 |
| SVM classification | Watermelon 3.0a | Linear/RBF SVM accuracy 0.7059, F1 0.6154 |
| UCI classification comparison | Wine | Linear SVM accuracy 0.9630, F1 0.9636 |
| WDBC breast cancer diagnosis | Breast Cancer WDBC | BP accuracy 0.9825, F1 0.9811; Linear SVM accuracy 0.9649, F1 0.9618 |
| SVR regression | Watermelon 3.0a | RBF-SVR RMSE 0.112893, MAE 0.095463 |

The corresponding `results/` directories retain per-sample predictions, evaluation metrics, and visual outputs for model-level and sample-level inspection.

## Key Strengths

- **From-scratch algorithm implementation**: Implemented logistic regression, linear discriminant analysis, and a C4.5-style gain-ratio decision tree, covering optimisation, discriminant functions, covariance estimation, gain-ratio splitting, and recursive tree construction.
- **Careful evaluation design**: Compared training-set resubstitution, LOOCV, stratified 5-fold CV, and bootstrap OOB on a small-sample dataset, reflecting attention to evaluation stability.
- **Applied benchmark comparison**: Used the WDBC breast cancer diagnosis dataset for benign-versus-malignant classification with 30 cytological features, comparing SVM, BP neural network, and a handwritten C4.5 decision tree.
- **Complete evaluation framework**: Beyond aggregate metrics, the projects retain prediction files, confusion matrices, decision boundaries, PR curves, and regression diagnostic plots, evaluating model performance through sample-level predictions, metric statistics, and visual evidence.

## Repository Structure

```text
.
+-- lab1/
|   +-- 3.3/      # Logistic regression experiments
|   +-- 3.5/      # Linear discriminant analysis experiments
+-- lab2/
|   +-- data/     # Watermelon, Wine, and WDBC datasets
|   +-- src/      # Experiment pipeline and handwritten C4.5 tree
|   +-- results/  # Metrics, predictions, models, and figures
|   +-- libsvm-3.31/
+-- README.md
```

Draft reports and Word documents are excluded through `.gitignore`; the repository keeps the code, datasets, and reproducible experimental artefacts.

## Environment

The projects use Python 3.10+ and common scientific-computing packages. `lab2` additionally relies on the bundled Windows LIBSVM executables.

```bash
pip install numpy matplotlib scikit-learn
```

Run representative experiments in `lab1`:

```bash
cd lab1/3.3
python src/logistic_regression_watermelon.py
python src/logistic_regression_quadratic_5fold.py

cd ../3.5
python src/linear_discriminant_analysis_watermelon.py
```

Run the full `lab2` experimental pipeline:

```bash
cd lab2
python src/run_lab2_experiments.py
```

