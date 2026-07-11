from __future__ import annotations

import argparse
import csv
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

from logistic_regression_loocv import calculate_metrics
from logistic_regression_watermelon import (
    DEFAULT_DATA_PATH,
    DEFAULT_LABEL,
    DEFAULT_POSITIVE_LABEL,
    LogisticRegression,
    load_dataset,
)


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


def bootstrap_oob_validate(
    row_ids: list[str],
    features: list[list[float]],
    labels: list[int],
    rounds: int,
    seed: int,
    max_iter: int,
    tolerance: float,
    l2: float,
) -> tuple[list[BootstrapRound], list[BootstrapPrediction], int]:
    if rounds <= 0:
        raise ValueError("rounds must be positive.")

    rng = random.Random(seed)
    sample_count = len(features)
    successful_rounds: list[BootstrapRound] = []
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

        model = LogisticRegression(max_iter=max_iter, tolerance=tolerance, l2=l2)
        try:
            model.fit(train_features, train_labels)
        except ValueError:
            skipped_rounds += 1
            continue

        probabilities = [model.predict_proba_one(row) for row in oob_features]
        predictions = [int(probability >= 0.5) for probability in probabilities]
        metrics = calculate_metrics(predictions, oob_labels)
        for oob_index, probability, prediction, label in zip(oob_indices, probabilities, predictions, oob_labels):
            oob_predictions.append(
                BootstrapPrediction(
                    round_index=round_index,
                    row_id=row_ids[oob_index],
                    probability=probability,
                    prediction=prediction,
                    label=label,
                )
            )
        successful_rounds.append(
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

    if not successful_rounds:
        raise ValueError("No valid bootstrap round was completed. Try increasing --rounds or --l2.")
    return successful_rounds, oob_predictions, skipped_rounds


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


def write_bootstrap_summary(
    path: Path,
    args: argparse.Namespace,
    feature_names: list[str],
    summary: dict[str, tuple[float, float]],
    completed_rounds: int,
    skipped_rounds: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# 西瓜数据集 3.0a 自助采样法评估结果

数据文件：`{args.data}`

特征列：{', '.join(feature_names)}

评估方式：自助采样法（bootstrap）袋外评估。每轮从 {completed_rounds and args.sample_count} 个样本中有放回抽取同样数量的训练样本，未被抽中的样本作为袋外测试样本。

随机种子：{args.seed}

计划轮数：{args.rounds}

有效轮数：{completed_rounds}

跳过轮数：{skipped_rounds}

优化方法：牛顿法；最大迭代次数：{args.max_iter}；收敛阈值：{args.tolerance}；L2 系数：{args.l2}

## 袋外评估指标（均值 ± 标准差）

| 指标 | 结果 |
|---|---:|
| 袋外样本数 | {summary['oob_size'][0]:.2f} ± {summary['oob_size'][1]:.2f} |
| 准确率 | {summary['accuracy'][0]:.4f} ± {summary['accuracy'][1]:.4f} |
| 精确率 | {summary['precision'][0]:.4f} ± {summary['precision'][1]:.4f} |
| 召回率 | {summary['recall'][0]:.4f} ± {summary['recall'][1]:.4f} |
| F1 | {summary['f1'][0]:.4f} ± {summary['f1'][1]:.4f} |
"""
    path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="使用自助采样法 bootstrap 袋外样本评估对数几率回归。")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="CSV 数据路径。")
    parser.add_argument("--label", default=DEFAULT_LABEL, help="标签列名。")
    parser.add_argument("--positive-label", default=DEFAULT_POSITIVE_LABEL, help="正类标签值。")
    parser.add_argument("--features", default=None, help="逗号分隔的特征列名；不填则优先使用“密度,含糖率”。")
    parser.add_argument("--rounds", type=int, default=200, help="自助采样轮数。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument("--max-iter", type=int, default=100, help="牛顿法最大迭代次数。")
    parser.add_argument("--tolerance", type=float, default=1e-8, help="停止迭代阈值。")
    parser.add_argument("--l2", type=float, default=0.01, help="L2 正则化系数；自助采样默认使用 0.01 提高重复抽样下的稳定性。")
    parser.add_argument("--round-output", type=Path, default=Path("results/自助采样/watermelon_3a_bootstrap_rounds.csv"), help="每轮袋外评估结果输出路径。")
    parser.add_argument("--prediction-output", type=Path, default=Path("results/自助采样/watermelon_3a_bootstrap_oob_predictions.csv"), help="袋外样本逐次预测结果输出路径。")
    parser.add_argument("--summary-output", type=Path, default=Path("results/自助采样/watermelon_3a_bootstrap_summary.md"), help="自助采样评估摘要输出路径。")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    row_ids, feature_names, features, labels = load_dataset(args.data, args.label, args.positive_label, args.features)
    args.sample_count = len(row_ids)
    rounds, oob_predictions, skipped_rounds = bootstrap_oob_validate(
        row_ids=row_ids,
        features=features,
        labels=labels,
        rounds=args.rounds,
        seed=args.seed,
        max_iter=args.max_iter,
        tolerance=args.tolerance,
        l2=args.l2,
    )
    summary = summarize_rounds(rounds)
    write_bootstrap_rounds(args.round_output, rounds)
    write_bootstrap_predictions(args.prediction_output, oob_predictions)
    write_bootstrap_summary(args.summary_output, args, feature_names, summary, len(rounds), skipped_rounds)

    print("=== 自助采样法袋外评估完成 ===")
    print(f"数据文件: {args.data}")
    print(f"特征列: {', '.join(feature_names)}")
    print(f"计划轮数: {args.rounds}; 有效轮数: {len(rounds)}; 跳过轮数: {skipped_rounds}")
    print(f"袋外样本数: {summary['oob_size'][0]:.2f} ± {summary['oob_size'][1]:.2f}")
    print(
        "OOB 指标: "
        f"accuracy={summary['accuracy'][0]:.4f}±{summary['accuracy'][1]:.4f}, "
        f"precision={summary['precision'][0]:.4f}±{summary['precision'][1]:.4f}, "
        f"recall={summary['recall'][0]:.4f}±{summary['recall'][1]:.4f}, "
        f"f1={summary['f1'][0]:.4f}±{summary['f1'][1]:.4f}"
    )
    print(f"每轮结果: {args.round_output}")
    print(f"袋外逐次预测: {args.prediction_output}")
    print(f"结果摘要: {args.summary_output}")


if __name__ == "__main__":
    main()
