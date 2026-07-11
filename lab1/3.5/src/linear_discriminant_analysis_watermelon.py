from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from dataclasses import dataclass
from typing import Iterable
from pathlib import Path


DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "watermelon_3a.csv"
DEFAULT_FEATURES = ("密度", "含糖率")
DEFAULT_LABEL = "好瓜"
DEFAULT_POSITIVE_LABEL = "是"
DEFAULT_BASE_OUTPUT = Path("results/线性判别分析")


def sigmoid(score: float) -> float:
    """Numerically stable sigmoid."""
    if score >= 0:
        exp_value = math.exp(-score)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(score)
    return exp_value / (1.0 + exp_value)


def dot(left: Iterable[float], right: Iterable[float]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve Ax=b by Gaussian elimination with partial pivoting."""
    size = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]

    for column in range(size):
        pivot_row = max(range(column, size), key=lambda row_index: abs(augmented[row_index][column]))
        pivot_value = augmented[pivot_row][column]
        if abs(pivot_value) < 1e-12:
            raise ValueError("Matrix is singular or ill-conditioned.")

        if pivot_row != column:
            augmented[column], augmented[pivot_row] = augmented[pivot_row], augmented[column]

        for row_index in range(column + 1, size):
            factor = augmented[row_index][column] / augmented[column][column]
            if factor == 0:
                continue
            for inner_column in range(column, size + 1):
                augmented[row_index][inner_column] -= factor * augmented[column][inner_column]

    solution = [0.0] * size
    for row_index in range(size - 1, -1, -1):
        known_sum = sum(augmented[row_index][column] * solution[column] for column in range(row_index + 1, size))
        solution[row_index] = (augmented[row_index][size] - known_sum) / augmented[row_index][row_index]
    return solution


def parse_feature_columns(header: list[str], label_column: str, user_features: str | None) -> list[str]:
    if user_features:
        feature_columns = [feature.strip() for feature in user_features.split(",") if feature.strip()]
    elif all(feature in header for feature in DEFAULT_FEATURES):
        feature_columns = list(DEFAULT_FEATURES)
    else:
        ignored_names = {label_column, "编号", "序号", "id", "ID", "index", "Index"}
        feature_columns = [column for column in header if column not in ignored_names]

    missing_features = [feature for feature in feature_columns if feature not in header]
    if missing_features:
        raise ValueError(f"Missing feature columns in CSV: {', '.join(missing_features)}")
    if label_column not in header:
        raise ValueError(f"Missing label column in CSV: {label_column}")
    return feature_columns


def load_dataset(
    path: Path,
    label_column: str,
    positive_label: str,
    feature_columns: str | None = None,
) -> tuple[list[str], list[str], list[list[float]], list[int]]:
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("CSV file has no header row.")
        selected_features = parse_feature_columns(reader.fieldnames, label_column, feature_columns)

        row_ids: list[str] = []
        features: list[list[float]] = []
        labels: list[int] = []
        for line_number, row in enumerate(reader, start=2):
            row_ids.append(row.get("编号", str(line_number - 1)))
            try:
                features.append([float(row[column]) for column in selected_features])
            except ValueError as exc:
                raise ValueError(f"Line {line_number} contains a non-numeric feature value.") from exc
            labels.append(1 if row[label_column].strip() == positive_label else 0)

    return row_ids, selected_features, features, labels


@dataclass
class LDAResult:
    coefficients: list[float]
    positive_mean: list[float]
    negative_mean: list[float]
    positive_prior: float
    negative_prior: float
    pooled_covariance: list[list[float]]


@dataclass
class BootstrapRound:
    round_index: int
    train_size: int
    oob_size: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int


@dataclass
class BootstrapPrediction:
    round_index: int
    row_id: str
    probability: float
    prediction: int
    label: int


@dataclass
class KFoldPrediction:
    fold_number: int
    row_id: str
    probability: float
    prediction: int
    label: int


@dataclass
class KFoldResult:
    fold_number: int
    train_size: int
    test_size: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int


class LinearDiscriminantAnalysis:
    """Binary LDA implemented from scratch with a shared covariance matrix."""

    def __init__(self, regularization: float = 1e-6) -> None:
        if regularization < 0:
            raise ValueError("regularization must be non-negative.")
        self.regularization = regularization
        self.result: LDAResult | None = None

    def fit(self, features: list[list[float]], labels: list[int]) -> LDAResult:
        if not features:
            raise ValueError("No training data was provided.")
        if len(features) != len(labels):
            raise ValueError("Feature rows and labels must have the same length.")
        if set(labels) - {0, 1}:
            raise ValueError("Labels must be encoded as 0 or 1.")
        feature_count = len(features[0])
        if any(len(row) != feature_count for row in features):
            raise ValueError("All feature rows must have the same length.")

        positive_rows = [row for row, label in zip(features, labels) if label == 1]
        negative_rows = [row for row, label in zip(features, labels) if label == 0]
        if not positive_rows or not negative_rows:
            raise ValueError("LDA requires both positive and negative samples.")

        positive_mean = mean_vector(positive_rows)
        negative_mean = mean_vector(negative_rows)
        pooled_covariance = pooled_covariance_matrix(
            positive_rows,
            negative_rows,
            positive_mean,
            negative_mean,
            self.regularization,
        )

        mean_difference = [pos - neg for pos, neg in zip(positive_mean, negative_mean)]
        weights = solve_linear_system(pooled_covariance, mean_difference)
        positive_prior = len(positive_rows) / len(features)
        negative_prior = len(negative_rows) / len(features)
        intercept = -0.5 * dot([pos + neg for pos, neg in zip(positive_mean, negative_mean)], weights)
        intercept += math.log(positive_prior / negative_prior)
        coefficients = [intercept] + weights
        self.result = LDAResult(
            coefficients=coefficients,
            positive_mean=positive_mean,
            negative_mean=negative_mean,
            positive_prior=positive_prior,
            negative_prior=negative_prior,
            pooled_covariance=pooled_covariance,
        )
        return self.result

    def decision_score_one(self, feature_row: list[float]) -> float:
        if self.result is None:
            raise ValueError("Model has not been fitted yet.")
        return self.result.coefficients[0] + dot(self.result.coefficients[1:], feature_row)

    def predict_proba_one(self, feature_row: list[float]) -> float:
        return sigmoid(self.decision_score_one(feature_row))

    def predict_one(self, feature_row: list[float], threshold: float = 0.5) -> int:
        return int(self.predict_proba_one(feature_row) >= threshold)


def mean_vector(rows: list[list[float]]) -> list[float]:
    return [sum(row[index] for row in rows) / len(rows) for index in range(len(rows[0]))]


def pooled_covariance_matrix(
    positive_rows: list[list[float]],
    negative_rows: list[list[float]],
    positive_mean: list[float],
    negative_mean: list[float],
    regularization: float,
) -> list[list[float]]:
    feature_count = len(positive_mean)
    scatter = [[0.0] * feature_count for _ in range(feature_count)]

    for rows, center in [(positive_rows, positive_mean), (negative_rows, negative_mean)]:
        for row in rows:
            centered = [value - mean for value, mean in zip(row, center)]
            for row_index in range(feature_count):
                for col_index in range(feature_count):
                    scatter[row_index][col_index] += centered[row_index] * centered[col_index]

    denominator = max(len(positive_rows) + len(negative_rows) - 2, 1)
    covariance = [[value / denominator for value in row] for row in scatter]
    for index in range(feature_count):
        covariance[index][index] += regularization
    return covariance


def calculate_metrics(predictions: list[int], labels: list[int]) -> dict[str, float | int]:
    true_positive = sum(pred == 1 and label == 1 for pred, label in zip(predictions, labels))
    true_negative = sum(pred == 0 and label == 0 for pred, label in zip(predictions, labels))
    false_positive = sum(pred == 1 and label == 0 for pred, label in zip(predictions, labels))
    false_negative = sum(pred == 0 and label == 1 for pred, label in zip(predictions, labels))
    accuracy = (true_positive + true_negative) / len(labels)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def write_prediction_csv(
    path: Path,
    row_ids: list[str],
    probabilities: list[float],
    predictions: list[int],
    labels: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["编号", "预测为好瓜的概率", "预测标签", "真实标签", "是否正确"])
        for row_id, probability, prediction, label in zip(row_ids, probabilities, predictions, labels):
            writer.writerow([row_id, f"{probability:.6f}", "是" if prediction else "否", "是" if label else "否", prediction == label])


def leave_one_out_validate(
    features: list[list[float]],
    labels: list[int],
    regularization: float,
) -> tuple[list[float], list[int]]:
    probabilities: list[float] = []
    predictions: list[int] = []
    for held_out_index in range(len(features)):
        train_features = [row for index, row in enumerate(features) if index != held_out_index]
        train_labels = [label for index, label in enumerate(labels) if index != held_out_index]
        model = LinearDiscriminantAnalysis(regularization=regularization)
        model.fit(train_features, train_labels)
        probability = model.predict_proba_one(features[held_out_index])
        probabilities.append(probability)
        predictions.append(int(probability >= 0.5))
    return probabilities, predictions


def stratified_kfold_indices(labels: list[int], fold_count: int, seed: int) -> list[list[int]]:
    if fold_count <= 1:
        raise ValueError("fold_count must be greater than 1.")
    if fold_count > len(labels):
        raise ValueError("fold_count cannot be greater than sample count.")

    rng = random.Random(seed)
    positive_indices = [index for index, label in enumerate(labels) if label == 1]
    negative_indices = [index for index, label in enumerate(labels) if label == 0]
    rng.shuffle(positive_indices)
    rng.shuffle(negative_indices)

    folds: list[list[int]] = [[] for _ in range(fold_count)]
    for class_indices in [positive_indices, negative_indices]:
        for offset, sample_index in enumerate(class_indices):
            folds[offset % fold_count].append(sample_index)

    return [sorted(fold) for fold in folds if fold]


def kfold_validate(
    row_ids: list[str],
    features: list[list[float]],
    labels: list[int],
    fold_count: int,
    seed: int,
    regularization: float,
) -> tuple[list[KFoldResult], list[KFoldPrediction]]:
    folds = stratified_kfold_indices(labels, fold_count, seed)
    fold_results: list[KFoldResult] = []
    fold_predictions: list[KFoldPrediction] = []

    for fold_number, test_indices in enumerate(folds, start=1):
        test_index_set = set(test_indices)
        train_indices = [index for index in range(len(features)) if index not in test_index_set]
        train_features = [features[index] for index in train_indices]
        train_labels = [labels[index] for index in train_indices]
        test_features = [features[index] for index in test_indices]
        test_labels = [labels[index] for index in test_indices]

        model = LinearDiscriminantAnalysis(regularization=regularization)
        model.fit(train_features, train_labels)
        probabilities = [model.predict_proba_one(row) for row in test_features]
        predictions = [int(probability >= 0.5) for probability in probabilities]
        metrics = calculate_metrics(predictions, test_labels)

        for test_index, probability, prediction, label in zip(test_indices, probabilities, predictions, test_labels):
            fold_predictions.append(
                KFoldPrediction(
                    fold_number=fold_number,
                    row_id=row_ids[test_index],
                    probability=probability,
                    prediction=prediction,
                    label=label,
                )
            )

        fold_results.append(
            KFoldResult(
                fold_number=fold_number,
                train_size=len(train_indices),
                test_size=len(test_indices),
                accuracy=float(metrics["accuracy"]),
                precision=float(metrics["precision"]),
                recall=float(metrics["recall"]),
                f1=float(metrics["f1"]),
                true_positive=int(metrics["true_positive"]),
                true_negative=int(metrics["true_negative"]),
                false_positive=int(metrics["false_positive"]),
                false_negative=int(metrics["false_negative"]),
            )
        )

    fold_predictions.sort(key=lambda prediction: int(prediction.row_id) if prediction.row_id.isdigit() else prediction.row_id)
    return fold_results, fold_predictions


def bootstrap_oob_validate(
    row_ids: list[str],
    features: list[list[float]],
    labels: list[int],
    rounds: int,
    seed: int,
    regularization: float,
) -> tuple[list[BootstrapRound], list[BootstrapPrediction], int]:
    if rounds <= 0:
        raise ValueError("rounds must be positive.")

    rng = random.Random(seed)
    sample_count = len(features)
    completed_rounds: list[BootstrapRound] = []
    oob_predictions: list[BootstrapPrediction] = []
    skipped_rounds = 0

    for round_index in range(1, rounds + 1):
        train_indices = [rng.randrange(sample_count) for _ in range(sample_count)]
        train_index_set = set(train_indices)
        oob_indices = [index for index in range(sample_count) if index not in train_index_set]
        train_labels = [labels[index] for index in train_indices]
        if not oob_indices or len(set(train_labels)) < 2:
            skipped_rounds += 1
            continue

        train_features = [features[index] for index in train_indices]
        oob_features = [features[index] for index in oob_indices]
        oob_labels = [labels[index] for index in oob_indices]

        try:
            model = LinearDiscriminantAnalysis(regularization=regularization)
            model.fit(train_features, train_labels)
        except ValueError:
            skipped_rounds += 1
            continue

        probabilities = [model.predict_proba_one(row) for row in oob_features]
        predictions = [int(probability >= 0.5) for probability in probabilities]
        metrics = calculate_metrics(predictions, oob_labels)

        for oob_index, probability, prediction, label in zip(oob_indices, probabilities, predictions, oob_labels):
            oob_predictions.append(
                BootstrapPrediction(round_index, row_ids[oob_index], probability, prediction, label)
            )

        completed_rounds.append(
            BootstrapRound(
                round_index=round_index,
                train_size=len(train_indices),
                oob_size=len(oob_indices),
                accuracy=float(metrics["accuracy"]),
                precision=float(metrics["precision"]),
                recall=float(metrics["recall"]),
                f1=float(metrics["f1"]),
                true_positive=int(metrics["true_positive"]),
                true_negative=int(metrics["true_negative"]),
                false_positive=int(metrics["false_positive"]),
                false_negative=int(metrics["false_negative"]),
            )
        )

    if not completed_rounds:
        raise ValueError("No valid bootstrap round was completed.")
    return completed_rounds, oob_predictions, skipped_rounds


def mean_and_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def summarize_rounds(rounds: list[BootstrapRound]) -> dict[str, tuple[float, float]]:
    return {
        "accuracy": mean_and_std([round_result.accuracy for round_result in rounds]),
        "precision": mean_and_std([round_result.precision for round_result in rounds]),
        "recall": mean_and_std([round_result.recall for round_result in rounds]),
        "f1": mean_and_std([round_result.f1 for round_result in rounds]),
        "oob_size": mean_and_std([float(round_result.oob_size) for round_result in rounds]),
    }


def write_bootstrap_rounds(path: Path, rounds: list[BootstrapRound]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["轮次", "训练样本数", "袋外样本数", "准确率", "精确率", "召回率", "F1", "TP", "TN", "FP", "FN"])
        for round_result in rounds:
            writer.writerow(
                [
                    round_result.round_index,
                    round_result.train_size,
                    round_result.oob_size,
                    f"{round_result.accuracy:.6f}",
                    f"{round_result.precision:.6f}",
                    f"{round_result.recall:.6f}",
                    f"{round_result.f1:.6f}",
                    round_result.true_positive,
                    round_result.true_negative,
                    round_result.false_positive,
                    round_result.false_negative,
                ]
            )


def write_bootstrap_predictions(path: Path, predictions: list[BootstrapPrediction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["轮次", "编号", "预测为好瓜的概率", "预测标签", "真实标签", "是否正确"])
        for prediction in predictions:
            writer.writerow(
                [
                    prediction.round_index,
                    prediction.row_id,
                    f"{prediction.probability:.6f}",
                    "是" if prediction.prediction else "否",
                    "是" if prediction.label else "否",
                    prediction.prediction == prediction.label,
                ]
            )


def write_kfold_predictions(path: Path, predictions: list[KFoldPrediction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["折数", "编号", "预测为好瓜的概率", "预测标签", "真实标签", "是否正确"])
        for prediction in predictions:
            writer.writerow(
                [
                    prediction.fold_number,
                    prediction.row_id,
                    f"{prediction.probability:.6f}",
                    "是" if prediction.prediction else "否",
                    "是" if prediction.label else "否",
                    prediction.prediction == prediction.label,
                ]
            )


def write_kfold_metrics(path: Path, fold_results: list[KFoldResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["折数", "训练样本数", "测试样本数", "准确率", "精确率", "召回率", "F1", "TP", "TN", "FP", "FN"])
        for result in fold_results:
            writer.writerow(
                [
                    result.fold_number,
                    result.train_size,
                    result.test_size,
                    f"{result.accuracy:.6f}",
                    f"{result.precision:.6f}",
                    f"{result.recall:.6f}",
                    f"{result.f1:.6f}",
                    result.true_positive,
                    result.true_negative,
                    result.false_positive,
                    result.false_negative,
                ]
            )


def format_vector(values: list[float]) -> str:
    return "(" + ", ".join(f"{value:.6f}" for value in values) + ")"


def write_training_summary(
    path: Path,
    args: argparse.Namespace,
    feature_names: list[str],
    result: LDAResult,
    metrics: dict[str, float | int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    weight_lines = [f"- 截距 b = {result.coefficients[0]:.6f}"]
    weight_lines.extend(
        f"- {feature} 的判别系数 w = {coefficient:.6f}"
        for feature, coefficient in zip(feature_names, result.coefficients[1:])
    )
    covariance_lines = [
        "| " + " | ".join(f"{value:.6f}" for value in row) + " |"
        for row in result.pooled_covariance
    ]
    content = f"""# 西瓜数据集 3.0a 线性判别分析训练集回代结果

数据文件：`{args.data}`

特征列：{', '.join(feature_names)}

类别先验：P(好瓜=是)={result.positive_prior:.4f}，P(好瓜=否)={result.negative_prior:.4f}

协方差正则化系数：{args.regularization}

## 类均值
- 好瓜均值向量：{format_vector(result.positive_mean)}
- 坏瓜均值向量：{format_vector(result.negative_mean)}

## 共享协方差矩阵
| {feature_names[0]} | {feature_names[1]} |
|---:|---:|
{chr(10).join(covariance_lines)}

## 判别函数
{chr(10).join(weight_lines)}

`P(好瓜=是|x)=σ({result.coefficients[0]:.6f} + {' + '.join(f'{coef:.6f}×{name}' for name, coef in zip(feature_names, result.coefficients[1:]))})`

## 训练集指标
- 准确率：{metrics['accuracy']:.4f}
- 精确率：{metrics['precision']:.4f}
- 召回率：{metrics['recall']:.4f}
- F1：{metrics['f1']:.4f}
- 混淆矩阵：TP={metrics['true_positive']}，TN={metrics['true_negative']}，FP={metrics['false_positive']}，FN={metrics['false_negative']}
"""
    path.write_text(content, encoding="utf-8")


def write_validation_summary(
    path: Path,
    title: str,
    args: argparse.Namespace,
    feature_names: list[str],
    metrics: dict[str, float | int],
    extra_lines: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# {title}

数据文件：`{args.data}`

特征列：{', '.join(feature_names)}

协方差正则化系数：{args.regularization}

{chr(10).join(extra_lines)}

## 评估指标
- 准确率：{metrics['accuracy']:.4f}
- 精确率：{metrics['precision']:.4f}
- 召回率：{metrics['recall']:.4f}
- F1：{metrics['f1']:.4f}
- 混淆矩阵：TP={metrics['true_positive']}，TN={metrics['true_negative']}，FP={metrics['false_positive']}，FN={metrics['false_negative']}
"""
    path.write_text(content, encoding="utf-8")


def write_kfold_summary(
    path: Path,
    args: argparse.Namespace,
    feature_names: list[str],
    fold_results: list[KFoldResult],
    metrics: dict[str, float | int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fold_lines = [
        "| 折数 | 训练样本数 | 测试样本数 | 准确率 | 精确率 | 召回率 | F1 | 混淆矩阵 |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    fold_lines.extend(
        (
            f"| {result.fold_number} | {result.train_size} | {result.test_size} | "
            f"{result.accuracy:.4f} | {result.precision:.4f} | {result.recall:.4f} | {result.f1:.4f} | "
            f"TP={result.true_positive}, TN={result.true_negative}, FP={result.false_positive}, FN={result.false_negative} |"
        )
        for result in fold_results
    )
    content = f"""# 西瓜数据集 3.0a 线性判别分析五折交叉验证结果

数据文件：`{args.data}`

特征列：{', '.join(feature_names)}

评估方式：分层 {args.fold_count} 折交叉验证。每一折使用约 4/5 样本训练，约 1/5 样本测试；每个样本只作为一次测试样本。

随机种子：{args.seed}

协方差正则化系数：{args.regularization}

## 总体指标
- 准确率：{metrics['accuracy']:.4f}
- 精确率：{metrics['precision']:.4f}
- 召回率：{metrics['recall']:.4f}
- F1：{metrics['f1']:.4f}
- 混淆矩阵：TP={metrics['true_positive']}，TN={metrics['true_negative']}，FP={metrics['false_positive']}，FN={metrics['false_negative']}

## 分折结果

{chr(10).join(fold_lines)}
"""
    path.write_text(content, encoding="utf-8")


def write_bootstrap_summary(
    path: Path,
    args: argparse.Namespace,
    feature_names: list[str],
    summary: dict[str, tuple[float, float]],
    completed_rounds: int,
    skipped_rounds: int,
    aggregate_metrics: dict[str, float | int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# 西瓜数据集 3.0a 线性判别分析自助采样法评估结果

数据文件：`{args.data}`

特征列：{', '.join(feature_names)}

评估方式：自助采样法（bootstrap）袋外评估。每轮从 {args.sample_count} 个样本中有放回抽取同样数量的训练样本，未被抽中的样本作为袋外测试样本。

随机种子：{args.seed}

计划轮数：{args.rounds}

有效轮数：{completed_rounds}

跳过轮数：{skipped_rounds}

协方差正则化系数：{args.regularization}

## 袋外评估指标（逐轮均值 ± 标准差）

| 指标 | 结果 |
|---|---:|
| 袋外样本数 | {summary['oob_size'][0]:.2f} ± {summary['oob_size'][1]:.2f} |
| 准确率 | {summary['accuracy'][0]:.4f} ± {summary['accuracy'][1]:.4f} |
| 精确率 | {summary['precision'][0]:.4f} ± {summary['precision'][1]:.4f} |
| 召回率 | {summary['recall'][0]:.4f} ± {summary['recall'][1]:.4f} |
| F1 | {summary['f1'][0]:.4f} ± {summary['f1'][1]:.4f} |

## 袋外逐次预测汇总指标
- 袋外逐次预测数：{sum(value for value in [aggregate_metrics['true_positive'], aggregate_metrics['true_negative'], aggregate_metrics['false_positive'], aggregate_metrics['false_negative']])}
- 准确率：{aggregate_metrics['accuracy']:.4f}
- 精确率：{aggregate_metrics['precision']:.4f}
- 召回率：{aggregate_metrics['recall']:.4f}
- F1：{aggregate_metrics['f1']:.4f}
- 混淆矩阵：TP={aggregate_metrics['true_positive']}，TN={aggregate_metrics['true_negative']}，FP={aggregate_metrics['false_positive']}，FN={aggregate_metrics['false_negative']}
"""
    path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从零实现二分类线性判别分析，并在西瓜数据集 3.0a 上评估。")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="CSV 数据路径。")
    parser.add_argument("--label", default=DEFAULT_LABEL, help="标签列名。")
    parser.add_argument("--positive-label", default=DEFAULT_POSITIVE_LABEL, help="正类标签值。")
    parser.add_argument("--features", default=None, help="逗号分隔的特征列名；不填则优先使用“密度,含糖率”。")
    parser.add_argument("--regularization", type=float, default=1e-6, help="共享协方差矩阵对角线正则化系数。")
    parser.add_argument("--rounds", type=int, default=200, help="自助采样轮数。")
    parser.add_argument("--seed", type=int, default=42, help="自助采样随机种子。")
    parser.add_argument("--fold-count", type=int, default=5, help="分层 K 折交叉验证折数。")
    parser.add_argument("--output-base", type=Path, default=DEFAULT_BASE_OUTPUT, help="LDA 结果输出根目录。")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    row_ids, feature_names, features, labels = load_dataset(args.data, args.label, args.positive_label, args.features)
    args.sample_count = len(row_ids)

    training_dir = args.output_base / "训练集回代"
    loocv_dir = args.output_base / "留一法"
    kfold_dir = args.output_base / "五折交叉验证"
    bootstrap_dir = args.output_base / "自助采样"

    model = LinearDiscriminantAnalysis(regularization=args.regularization)
    result = model.fit(features, labels)
    training_probabilities = [model.predict_proba_one(row) for row in features]
    training_predictions = [int(probability >= 0.5) for probability in training_probabilities]
    training_metrics = calculate_metrics(training_predictions, labels)
    write_prediction_csv(training_dir / "watermelon_3a_lda_predictions.csv", row_ids, training_probabilities, training_predictions, labels)
    write_training_summary(training_dir / "watermelon_3a_lda_summary.md", args, feature_names, result, training_metrics)

    loocv_probabilities, loocv_predictions = leave_one_out_validate(features, labels, args.regularization)
    loocv_metrics = calculate_metrics(loocv_predictions, labels)
    write_prediction_csv(loocv_dir / "watermelon_3a_lda_loocv_predictions.csv", row_ids, loocv_probabilities, loocv_predictions, labels)
    write_validation_summary(
        loocv_dir / "watermelon_3a_lda_loocv_summary.md",
        "西瓜数据集 3.0a 线性判别分析留一法交叉验证结果",
        args,
        feature_names,
        loocv_metrics,
        [f"验证方式：留一法交叉验证。每次取 1 个样本作为验证样本，其余 {len(row_ids) - 1} 个样本作为训练集，共训练 {len(row_ids)} 个模型。"],
    )

    fold_results, fold_predictions = kfold_validate(
        row_ids=row_ids,
        features=features,
        labels=labels,
        fold_count=args.fold_count,
        seed=args.seed,
        regularization=args.regularization,
    )
    kfold_predictions = [prediction.prediction for prediction in fold_predictions]
    kfold_labels = [prediction.label for prediction in fold_predictions]
    kfold_metrics = calculate_metrics(kfold_predictions, kfold_labels)
    write_kfold_predictions(kfold_dir / "watermelon_3a_lda_5fold_predictions.csv", fold_predictions)
    write_kfold_metrics(kfold_dir / "watermelon_3a_lda_5fold_metrics.csv", fold_results)
    write_kfold_summary(
        kfold_dir / "watermelon_3a_lda_5fold_summary.md",
        args,
        feature_names,
        fold_results,
        kfold_metrics,
    )

    rounds, oob_predictions, skipped_rounds = bootstrap_oob_validate(
        row_ids=row_ids,
        features=features,
        labels=labels,
        rounds=args.rounds,
        seed=args.seed,
        regularization=args.regularization,
    )
    bootstrap_summary = summarize_rounds(rounds)
    write_bootstrap_rounds(bootstrap_dir / "watermelon_3a_lda_bootstrap_rounds.csv", rounds)
    write_bootstrap_predictions(bootstrap_dir / "watermelon_3a_lda_bootstrap_oob_predictions.csv", oob_predictions)
    aggregate_predictions = [prediction.prediction for prediction in oob_predictions]
    aggregate_labels = [prediction.label for prediction in oob_predictions]
    aggregate_metrics = calculate_metrics(aggregate_predictions, aggregate_labels)
    write_bootstrap_summary(
        bootstrap_dir / "watermelon_3a_lda_bootstrap_summary.md",
        args,
        feature_names,
        bootstrap_summary,
        len(rounds),
        skipped_rounds,
        aggregate_metrics,
    )

    print("=== 线性判别分析实验完成 ===")
    print(f"数据文件: {args.data}")
    print(f"特征列: {', '.join(feature_names)}")
    print(f"判别函数: {result.coefficients[0]:.6f} + " + " + ".join(f"{coef:.6f}*{name}" for name, coef in zip(feature_names, result.coefficients[1:])))
    print(
        "训练集回代: "
        f"accuracy={training_metrics['accuracy']:.4f}, precision={training_metrics['precision']:.4f}, "
        f"recall={training_metrics['recall']:.4f}, f1={training_metrics['f1']:.4f}"
    )
    print(
        "LOOCV: "
        f"accuracy={loocv_metrics['accuracy']:.4f}, precision={loocv_metrics['precision']:.4f}, "
        f"recall={loocv_metrics['recall']:.4f}, f1={loocv_metrics['f1']:.4f}"
    )
    print(
        "5-fold CV: "
        f"accuracy={kfold_metrics['accuracy']:.4f}, precision={kfold_metrics['precision']:.4f}, "
        f"recall={kfold_metrics['recall']:.4f}, f1={kfold_metrics['f1']:.4f}"
    )
    print(
        "Bootstrap OOB: "
        f"accuracy={aggregate_metrics['accuracy']:.4f}, precision={aggregate_metrics['precision']:.4f}, "
        f"recall={aggregate_metrics['recall']:.4f}, f1={aggregate_metrics['f1']:.4f}"
    )
    print(f"结果目录: {args.output_base}")


if __name__ == "__main__":
    main()
