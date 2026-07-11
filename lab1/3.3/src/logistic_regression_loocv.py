from __future__ import annotations

import argparse
import csv
from pathlib import Path

from logistic_regression_watermelon import (
    DEFAULT_DATA_PATH,
    DEFAULT_LABEL,
    DEFAULT_POSITIVE_LABEL,
    LogisticRegression,
    load_dataset,
)


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


def leave_one_out_validate(
    features: list[list[float]],
    labels: list[int],
    max_iter: int,
    tolerance: float,
    l2: float,
) -> tuple[list[float], list[int], list[int]]:
    probabilities: list[float] = []
    predictions: list[int] = []
    iterations: list[int] = []

    for held_out_index in range(len(features)):
        train_features = [row for index, row in enumerate(features) if index != held_out_index]
        train_labels = [label for index, label in enumerate(labels) if index != held_out_index]
        model = LogisticRegression(max_iter=max_iter, tolerance=tolerance, l2=l2)
        result = model.fit(train_features, train_labels)
        probability = model.predict_proba_one(features[held_out_index])
        probabilities.append(probability)
        predictions.append(int(probability >= 0.5))
        iterations.append(result.iterations)

    return probabilities, predictions, iterations


def write_loocv_predictions(
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


def write_loocv_summary(
    path: Path,
    args: argparse.Namespace,
    feature_names: list[str],
    metrics: dict[str, float | int],
    iterations: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    average_iterations = sum(iterations) / len(iterations)
    summary = f"""# 西瓜数据集 3.0a 留一法交叉验证结果

数据文件：`{args.data}`

特征列：{', '.join(feature_names)}

验证方式：留一法交叉验证（LOOCV）。每次取 1 个样本作为验证样本，其余 {len(iterations) - 1} 个样本作为训练集，共训练 {len(iterations)} 个模型。

优化方法：牛顿法；最大迭代次数：{args.max_iter}；收敛阈值：{args.tolerance}；L2 系数：{args.l2}

## 交叉验证指标
- 准确率：{metrics['accuracy']:.4f}
- 精确率：{metrics['precision']:.4f}
- 召回率：{metrics['recall']:.4f}
- F1：{metrics['f1']:.4f}
- 混淆矩阵：TP={metrics['true_positive']}，TN={metrics['true_negative']}，FP={metrics['false_positive']}，FN={metrics['false_negative']}
- 平均迭代次数：{average_iterations:.2f}
"""
    path.write_text(summary, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="使用留一法交叉验证评估二分类对数几率回归。")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="CSV 数据路径。")
    parser.add_argument("--label", default=DEFAULT_LABEL, help="标签列名。")
    parser.add_argument("--positive-label", default=DEFAULT_POSITIVE_LABEL, help="正类标签值。")
    parser.add_argument("--features", default=None, help="逗号分隔的特征列名；不填则优先使用“密度,含糖率”。")
    parser.add_argument("--max-iter", type=int, default=100, help="牛顿法最大迭代次数。")
    parser.add_argument("--tolerance", type=float, default=1e-8, help="停止迭代阈值。")
    parser.add_argument("--l2", type=float, default=0.0, help="L2 正则化系数；默认不使用正则化。")
    parser.add_argument("--prediction-output", type=Path, default=Path("results/留一法/watermelon_3a_loocv_predictions.csv"), help="留一法逐样本预测输出路径。")
    parser.add_argument("--summary-output", type=Path, default=Path("results/留一法/watermelon_3a_loocv_summary.md"), help="留一法结果摘要输出路径。")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    row_ids, feature_names, features, labels = load_dataset(args.data, args.label, args.positive_label, args.features)
    probabilities, predictions, iterations = leave_one_out_validate(features, labels, args.max_iter, args.tolerance, args.l2)
    metrics = calculate_metrics(predictions, labels)
    write_loocv_predictions(args.prediction_output, row_ids, probabilities, predictions, labels)
    write_loocv_summary(args.summary_output, args, feature_names, metrics, iterations)

    print("=== 留一法交叉验证完成 ===")
    print(f"数据文件: {args.data}")
    print(f"特征列: {', '.join(feature_names)}")
    print(
        "LOOCV 指标: "
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
