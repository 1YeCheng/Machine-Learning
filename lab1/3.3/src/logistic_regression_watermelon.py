from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "watermelon_3a.csv"
DEFAULT_FEATURES = ("密度", "含糖率")
DEFAULT_LABEL = "好瓜"
DEFAULT_POSITIVE_LABEL = "是"


def sigmoid(score: float) -> float:
    """Numerically stable sigmoid."""
    if score >= 0:
        exp_value = math.exp(-score)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(score)
    return exp_value / (1.0 + exp_value)


def softplus(score: float) -> float:
    """Stable log(1 + exp(score))."""
    if score > 0:
        return score + math.log1p(math.exp(-score))
    return math.log1p(math.exp(score))


def dot(left: Iterable[float], right: Iterable[float]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def vector_norm(values: Iterable[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve Ax=b by Gaussian elimination with partial pivoting."""
    size = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]

    for column in range(size):
        pivot_row = max(range(column, size), key=lambda row_index: abs(augmented[row_index][column]))
        pivot_value = augmented[pivot_row][column]
        if abs(pivot_value) < 1e-12:
            raise ValueError("Hessian is singular or ill-conditioned.")

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


@dataclass
class TrainingResult:
    coefficients: list[float]
    loss: float
    iterations: int
    converged: bool


class LogisticRegression:
    """Binary logistic regression implemented from scratch with Newton's method."""

    def __init__(self, max_iter: int = 100, tolerance: float = 1e-8, l2: float = 0.0) -> None:
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive.")
        if l2 < 0:
            raise ValueError("l2 must be non-negative.")
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.l2 = l2
        self.coefficients: list[float] = []

    def fit(self, features: list[list[float]], labels: list[int]) -> TrainingResult:
        if not features:
            raise ValueError("No training data was provided.")
        if len(features) != len(labels):
            raise ValueError("Feature rows and labels must have the same length.")
        feature_count = len(features[0])
        if any(len(row) != feature_count for row in features):
            raise ValueError("All feature rows must have the same length.")
        if set(labels) - {0, 1}:
            raise ValueError("Labels must be encoded as 0 or 1.")

        design_matrix = [[1.0] + row for row in features]
        parameter_count = feature_count + 1
        coefficients = [0.0] * parameter_count
        converged = False

        for iteration in range(1, self.max_iter + 1):
            probabilities = [sigmoid(dot(coefficients, row)) for row in design_matrix]
            gradient = [0.0] * parameter_count
            hessian = [[0.0] * parameter_count for _ in range(parameter_count)]

            for row, probability, label in zip(design_matrix, probabilities, labels):
                residual = probability - label
                weight = probability * (1.0 - probability)
                for col_index in range(parameter_count):
                    gradient[col_index] += residual * row[col_index]
                    for inner_index in range(parameter_count):
                        hessian[col_index][inner_index] += weight * row[col_index] * row[inner_index]

            for index in range(1, parameter_count):
                gradient[index] += self.l2 * coefficients[index]
                hessian[index][index] += self.l2

            step = solve_linear_system(hessian, gradient)
            coefficients = [coef - delta for coef, delta in zip(coefficients, step)]

            if vector_norm(step) < self.tolerance:
                converged = True
                break

        self.coefficients = coefficients
        final_loss = self.loss(features, labels)
        return TrainingResult(coefficients, final_loss, iteration, converged)

    def predict_proba_one(self, feature_row: list[float]) -> float:
        if not self.coefficients:
            raise ValueError("Model has not been fitted yet.")
        return sigmoid(self.coefficients[0] + dot(self.coefficients[1:], feature_row))

    def predict_one(self, feature_row: list[float], threshold: float = 0.5) -> int:
        return int(self.predict_proba_one(feature_row) >= threshold)

    def loss(self, features: list[list[float]], labels: list[int]) -> float:
        design_matrix = [[1.0] + row for row in features]
        negative_log_likelihood = 0.0
        for row, label in zip(design_matrix, labels):
            score = dot(self.coefficients, row)
            negative_log_likelihood += softplus(score) - label * score
        regularization = 0.5 * self.l2 * sum(coef * coef for coef in self.coefficients[1:])
        return negative_log_likelihood + regularization


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
            label_value = row[label_column].strip()
            labels.append(1 if label_value == positive_label else 0)

    return row_ids, selected_features, features, labels


def evaluate(model: LogisticRegression, features: list[list[float]], labels: list[int]) -> dict[str, float | int]:
    predictions = [model.predict_one(row) for row in features]
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


def write_summary(path: Path, args: argparse.Namespace, feature_names: list[str], result: TrainingResult, metrics: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    coefficient_lines = [f"- 截距 b = {result.coefficients[0]:.6f}"]
    coefficient_lines.extend(
        f"- {feature} 的权重 w = {coefficient:.6f}"
        for feature, coefficient in zip(feature_names, result.coefficients[1:])
    )
    summary = f"""# 西瓜数据集 3.0a 对数几率回归结果

数据文件：`{args.data}`

特征列：{', '.join(feature_names)}

优化方法：牛顿法；最大迭代次数：{args.max_iter}；收敛阈值：{args.tolerance}；L2 系数：{args.l2}

## 模型参数
{chr(10).join(coefficient_lines)}

决策函数：

`P(好瓜=是|x)=σ({result.coefficients[0]:.6f} + {' + '.join(f'{coef:.6f}×{name}' for name, coef in zip(feature_names, result.coefficients[1:]))})`

## 训练集指标
- 是否收敛：{result.converged}
- 迭代次数：{result.iterations}
- 负对数似然：{result.loss:.6f}
- 准确率：{metrics['accuracy']:.4f}
- 精确率：{metrics['precision']:.4f}
- 召回率：{metrics['recall']:.4f}
- F1：{metrics['f1']:.4f}
- 混淆矩阵：TP={metrics['true_positive']}，TN={metrics['true_negative']}，FP={metrics['false_positive']}，FN={metrics['false_negative']}
"""
    path.write_text(summary, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从零实现二分类对数几率回归，并在 CSV 数据集上训练与评估。")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="CSV 数据路径。")
    parser.add_argument("--label", default=DEFAULT_LABEL, help="标签列名。")
    parser.add_argument("--positive-label", default=DEFAULT_POSITIVE_LABEL, help="正类标签值。")
    parser.add_argument("--features", default=None, help="逗号分隔的特征列名；不填则优先使用“密度,含糖率”。")
    parser.add_argument("--max-iter", type=int, default=100, help="牛顿法最大迭代次数。")
    parser.add_argument("--tolerance", type=float, default=1e-8, help="停止迭代阈值。")
    parser.add_argument("--l2", type=float, default=0.0, help="L2 正则化系数；默认不使用正则化。")
    parser.add_argument("--prediction-output", type=Path, default=Path("results/训练集回代/watermelon_3a_predictions.csv"), help="逐样本预测结果输出路径。")
    parser.add_argument("--summary-output", type=Path, default=Path("results/训练集回代/watermelon_3a_summary.md"), help="结果摘要输出路径。")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    row_ids, feature_names, features, labels = load_dataset(args.data, args.label, args.positive_label, args.features)

    model = LogisticRegression(max_iter=args.max_iter, tolerance=args.tolerance, l2=args.l2)
    result = model.fit(features, labels)
    metrics = evaluate(model, features, labels)
    probabilities = [model.predict_proba_one(row) for row in features]
    predictions = [int(probability >= 0.5) for probability in probabilities]

    write_prediction_csv(args.prediction_output, row_ids, probabilities, predictions, labels)
    write_summary(args.summary_output, args, feature_names, result, metrics)

    print("=== 对数几率回归训练完成 ===")
    print(f"数据文件: {args.data}")
    print(f"特征列: {', '.join(feature_names)}")
    print(f"是否收敛: {result.converged}; 迭代次数: {result.iterations}; 负对数似然: {result.loss:.6f}")
    print(f"截距 b: {result.coefficients[0]:.6f}")
    for feature_name, coefficient in zip(feature_names, result.coefficients[1:]):
        print(f"权重 {feature_name}: {coefficient:.6f}")
    print(
        "训练集指标: "
        f"accuracy={metrics['accuracy']:.4f}, precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}"
    )
    print(
        "混淆矩阵: "
        f"TP={metrics['true_positive']}, TN={metrics['true_negative']}, "
        f"FP={metrics['false_positive']}, FN={metrics['false_negative']}"
    )
    print(f"逐样本预测结果: {args.prediction_output}")
    print(f"结果摘要: {args.summary_output}")


if __name__ == "__main__":
    main()
