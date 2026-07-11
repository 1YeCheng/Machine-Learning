from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

from logistic_regression_loocv import calculate_metrics
from logistic_regression_quadratic import expand_quadratic_features
from logistic_regression_watermelon import (
    DEFAULT_DATA_PATH,
    DEFAULT_LABEL,
    DEFAULT_POSITIVE_LABEL,
    LogisticRegression,
    load_dataset,
)


@dataclass
class FoldResult:
    fold: int
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


def stratified_kfold_indices(labels: list[int], fold_count: int, seed: int) -> list[list[int]]:
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2.")
    if fold_count > len(labels):
        raise ValueError("fold_count must not exceed sample count.")

    rng = random.Random(seed)
    positive_indices = [index for index, label in enumerate(labels) if label == 1]
    negative_indices = [index for index, label in enumerate(labels) if label == 0]
    rng.shuffle(positive_indices)
    rng.shuffle(negative_indices)

    folds = [[] for _ in range(fold_count)]
    for group in [positive_indices, negative_indices]:
        for position, index in enumerate(group):
            folds[position % fold_count].append(index)

    for fold in folds:
        fold.sort()
    return folds


def run_kfold_validation(
    row_ids: list[str],
    features: list[list[float]],
    labels: list[int],
    fold_count: int,
    seed: int,
    max_iter: int,
    tolerance: float,
    l2: float,
) -> tuple[list[FoldResult], list[list[str | int | float | bool]]]:
    folds = stratified_kfold_indices(labels, fold_count, seed)
    fold_results: list[FoldResult] = []
    prediction_rows: list[list[str | int | float | bool]] = []

    all_predictions: list[int] = []
    all_labels: list[int] = []

    for fold_number, test_indices in enumerate(folds, start=1):
        test_index_set = set(test_indices)
        train_indices = [index for index in range(len(features)) if index not in test_index_set]
        train_features = [features[index] for index in train_indices]
        train_labels = [labels[index] for index in train_indices]
        test_features = [features[index] for index in test_indices]
        test_labels = [labels[index] for index in test_indices]

        model = LogisticRegression(max_iter=max_iter, tolerance=tolerance, l2=l2)
        model.fit(train_features, train_labels)

        probabilities = [model.predict_proba_one(row) for row in test_features]
        predictions = [int(probability >= 0.5) for probability in probabilities]
        metrics = calculate_metrics(predictions, test_labels)

        all_predictions.extend(predictions)
        all_labels.extend(test_labels)
        fold_results.append(
            FoldResult(
                fold=fold_number,
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

        for index, probability, prediction, label in zip(test_indices, probabilities, predictions, test_labels):
            prediction_rows.append(
                [
                    fold_number,
                    row_ids[index],
                    f"{probability:.6f}",
                    "是" if prediction else "否",
                    "是" if label else "否",
                    prediction == label,
                ]
            )

    overall_metrics = calculate_metrics(all_predictions, all_labels)
    fold_results.append(
        FoldResult(
            fold=0,
            train_size=0,
            test_size=len(labels),
            accuracy=float(overall_metrics["accuracy"]),
            precision=float(overall_metrics["precision"]),
            recall=float(overall_metrics["recall"]),
            f1=float(overall_metrics["f1"]),
            true_positive=int(overall_metrics["true_positive"]),
            true_negative=int(overall_metrics["true_negative"]),
            false_positive=int(overall_metrics["false_positive"]),
            false_negative=int(overall_metrics["false_negative"]),
        )
    )
    return fold_results, prediction_rows


def write_fold_results(path: Path, fold_results: list[FoldResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["折数", "训练样本数", "测试样本数", "准确率", "精确率", "召回率", "F1", "TP", "TN", "FP", "FN"])
        for result in fold_results:
            fold_name = "总体" if result.fold == 0 else result.fold
            writer.writerow(
                [
                    fold_name,
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


def write_predictions(path: Path, prediction_rows: list[list[str | int | float | bool]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["折数", "编号", "预测为好瓜的概率", "预测标签", "真实标签", "是否正确"])
        writer.writerows(prediction_rows)


def write_summary(
    path: Path,
    args: argparse.Namespace,
    expanded_feature_names: list[str],
    fold_results: list[FoldResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    overall = fold_results[-1]
    fold_lines = []
    for result in fold_results[:-1]:
        fold_lines.append(
            f"| {result.fold} | {result.train_size} | {result.test_size} | {result.accuracy:.4f} | "
            f"{result.precision:.4f} | {result.recall:.4f} | {result.f1:.4f} | "
            f"TP={result.true_positive}, TN={result.true_negative}, FP={result.false_positive}, FN={result.false_negative} |"
        )

    content = f"""# 西瓜数据集 3.0a 二次特征五折交叉验证结果

数据文件：`{args.data}`

扩展特征列：{', '.join(expanded_feature_names)}

评估方式：分层 5 折交叉验证。每一折使用约 4/5 样本训练，约 1/5 样本测试；每个样本只作为一次测试样本。

随机种子：{args.seed}

L2 系数：{args.l2}

## 总体指标

| 指标 | 结果 |
|---|---:|
| 准确率 | {overall.accuracy:.4f} |
| 精确率 | {overall.precision:.4f} |
| 召回率 | {overall.recall:.4f} |
| F1 | {overall.f1:.4f} |
| 混淆矩阵 | TP={overall.true_positive}，TN={overall.true_negative}，FP={overall.false_positive}，FN={overall.false_negative} |

## 分折结果

| 折数 | 训练样本数 | 测试样本数 | 准确率 | 精确率 | 召回率 | F1 | 混淆矩阵 |
|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(fold_lines)}
"""
    path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="使用 5 折交叉验证评估二次项和交叉项对数几率回归。")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="CSV 数据路径。")
    parser.add_argument("--label", default=DEFAULT_LABEL, help="标签列名。")
    parser.add_argument("--positive-label", default=DEFAULT_POSITIVE_LABEL, help="正类标签值。")
    parser.add_argument("--features", default=None, help="两个逗号分隔的基础特征列名；不填则优先使用“密度,含糖率”。")
    parser.add_argument("--folds", type=int, default=5, help="交叉验证折数。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument("--max-iter", type=int, default=100, help="牛顿法最大迭代次数。")
    parser.add_argument("--tolerance", type=float, default=1e-8, help="停止迭代阈值。")
    parser.add_argument("--l2", type=float, default=0.01, help="L2 正则化系数。")
    parser.add_argument("--fold-output", type=Path, default=Path("results/二次特征/watermelon_3a_quadratic_5fold_metrics.csv"), help="分折指标输出路径。")
    parser.add_argument("--prediction-output", type=Path, default=Path("results/二次特征/watermelon_3a_quadratic_5fold_predictions.csv"), help="五折逐样本预测输出路径。")
    parser.add_argument("--summary-output", type=Path, default=Path("results/二次特征/watermelon_3a_quadratic_5fold_summary.md"), help="五折结果摘要输出路径。")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    row_ids, base_feature_names, base_features, labels = load_dataset(args.data, args.label, args.positive_label, args.features)
    expanded_feature_names, expanded_features = expand_quadratic_features(base_feature_names, base_features)
    fold_results, prediction_rows = run_kfold_validation(
        row_ids=row_ids,
        features=expanded_features,
        labels=labels,
        fold_count=args.folds,
        seed=args.seed,
        max_iter=args.max_iter,
        tolerance=args.tolerance,
        l2=args.l2,
    )
    write_fold_results(args.fold_output, fold_results)
    write_predictions(args.prediction_output, prediction_rows)
    write_summary(args.summary_output, args, expanded_feature_names, fold_results)

    overall = fold_results[-1]
    print("=== 二次特征 5 折交叉验证完成 ===")
    print(f"基础特征列: {', '.join(base_feature_names)}")
    print(f"扩展特征列: {', '.join(expanded_feature_names)}")
    print(
        "5-fold 指标: "
        f"accuracy={overall.accuracy:.4f}, precision={overall.precision:.4f}, "
        f"recall={overall.recall:.4f}, f1={overall.f1:.4f}"
    )
    print(
        "混淆矩阵: "
        f"TP={overall.true_positive}, TN={overall.true_negative}, "
        f"FP={overall.false_positive}, FN={overall.false_negative}"
    )
    print(f"分折指标: {args.fold_output}")
    print(f"逐样本预测: {args.prediction_output}")
    print(f"结果摘要: {args.summary_output}")


if __name__ == "__main__":
    main()
