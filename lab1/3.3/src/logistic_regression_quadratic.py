from __future__ import annotations

import argparse
from pathlib import Path

from logistic_regression_watermelon import (
    DEFAULT_DATA_PATH,
    DEFAULT_LABEL,
    DEFAULT_POSITIVE_LABEL,
    LogisticRegression,
    evaluate,
    load_dataset,
    write_prediction_csv,
    write_summary,
)


def expand_quadratic_features(
    feature_names: list[str],
    features: list[list[float]],
) -> tuple[list[str], list[list[float]]]:
    if len(feature_names) != 2:
        raise ValueError("二次型扩展脚本当前要求恰好输入两个基础特征。")

    first_name, second_name = feature_names
    expanded_names = [
        first_name,
        second_name,
        f"{first_name}^2",
        f"{second_name}^2",
        f"{first_name}×{second_name}",
    ]
    expanded_features = []
    for first_value, second_value in features:
        expanded_features.append(
            [
                first_value,
                second_value,
                first_value * first_value,
                second_value * second_value,
                first_value * second_value,
            ]
        )
    return expanded_names, expanded_features


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="加入二次项和交叉项的对数几率回归。")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="CSV 数据路径。")
    parser.add_argument("--label", default=DEFAULT_LABEL, help="标签列名。")
    parser.add_argument("--positive-label", default=DEFAULT_POSITIVE_LABEL, help="正类标签值。")
    parser.add_argument("--features", default=None, help="两个逗号分隔的基础特征列名；不填则优先使用“密度,含糖率”。")
    parser.add_argument("--max-iter", type=int, default=100, help="牛顿法最大迭代次数。")
    parser.add_argument("--tolerance", type=float, default=1e-8, help="停止迭代阈值。")
    parser.add_argument("--l2", type=float, default=0.01, help="L2 正则化系数；二次特征默认使用 0.01 抑制小样本过拟合。")
    parser.add_argument("--prediction-output", type=Path, default=Path("results/二次特征/watermelon_3a_quadratic_predictions.csv"), help="二次特征逐样本预测输出路径。")
    parser.add_argument("--summary-output", type=Path, default=Path("results/二次特征/watermelon_3a_quadratic_summary.md"), help="二次特征结果摘要输出路径。")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    row_ids, base_feature_names, base_features, labels = load_dataset(args.data, args.label, args.positive_label, args.features)
    expanded_feature_names, expanded_features = expand_quadratic_features(base_feature_names, base_features)

    model = LogisticRegression(max_iter=args.max_iter, tolerance=args.tolerance, l2=args.l2)
    try:
        result = model.fit(expanded_features, labels)
    except ValueError as exc:
        if args.l2 == 0:
            raise SystemExit("二次特征下 Hessian 可能病态；请使用 --l2 0.01 或更大的正则化系数后重试。") from exc
        raise
    metrics = evaluate(model, expanded_features, labels)
    probabilities = [model.predict_proba_one(row) for row in expanded_features]
    predictions = [int(probability >= 0.5) for probability in probabilities]

    write_prediction_csv(args.prediction_output, row_ids, probabilities, predictions, labels)
    write_summary(args.summary_output, args, expanded_feature_names, result, metrics)

    print("=== 二次特征对数几率回归训练完成 ===")
    print(f"数据文件: {args.data}")
    print(f"基础特征列: {', '.join(base_feature_names)}")
    print(f"扩展特征列: {', '.join(expanded_feature_names)}")
    print(f"是否收敛: {result.converged}; 迭代次数: {result.iterations}; 负对数似然: {result.loss:.6f}")
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
