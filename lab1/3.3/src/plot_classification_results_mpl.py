from __future__ import annotations

import argparse
import csv
import warnings
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings("ignore", message="Pandas requires version .*", category=UserWarning)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager


@dataclass
class PredictionRow:
    probability: float
    prediction: int
    label: int


@dataclass
class Metrics:
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    auprc: float


def configure_plot_style() -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected_font = None
    for font_name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]:
        if font_name in available_fonts:
            selected_font = font_name
            break
    sns.set_theme(style="whitegrid", context="notebook")
    if selected_font:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [selected_font]
    plt.rcParams["axes.unicode_minus"] = False


def load_predictions(path: Path) -> list[PredictionRow]:
    rows: list[PredictionRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"预测为好瓜的概率", "预测标签", "真实标签"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"Prediction CSV must contain columns: {', '.join(sorted(required_columns))}")
        for row in reader:
            rows.append(
                PredictionRow(
                    probability=float(row["预测为好瓜的概率"]),
                    prediction=1 if row["预测标签"].strip() == "是" else 0,
                    label=1 if row["真实标签"].strip() == "是" else 0,
                )
            )
    if not rows:
        raise ValueError("Prediction CSV is empty.")
    return rows


def precision_recall_curve(rows: list[PredictionRow]) -> tuple[list[tuple[float, float]], float]:
    sorted_rows = sorted(rows, key=lambda row: row.probability, reverse=True)
    positive_count = sum(row.label == 1 for row in sorted_rows)
    if positive_count == 0:
        raise ValueError("PR curve requires at least one positive sample.")

    points = [(0.0, 1.0)]
    true_positive = 0
    false_positive = 0
    auprc = 0.0
    previous_recall = 0.0

    for row in sorted_rows:
        if row.label == 1:
            true_positive += 1
        else:
            false_positive += 1
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / positive_count
        points.append((recall, precision))
        if recall > previous_recall:
            auprc += precision * (recall - previous_recall)
            previous_recall = recall

    return points, auprc


def calculate_metrics(rows: list[PredictionRow], auprc: float) -> Metrics:
    true_positive = sum(row.prediction == 1 and row.label == 1 for row in rows)
    true_negative = sum(row.prediction == 0 and row.label == 0 for row in rows)
    false_positive = sum(row.prediction == 1 and row.label == 0 for row in rows)
    false_negative = sum(row.prediction == 0 and row.label == 1 for row in rows)
    accuracy = (true_positive + true_negative) / len(rows)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Metrics(true_positive, true_negative, false_positive, false_negative, accuracy, precision, recall, f1, auprc)


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(metrics: Metrics, output_path: Path) -> None:
    matrix = [
        [metrics.true_positive, metrics.false_negative],
        [metrics.false_positive, metrics.true_negative],
    ]
    annotations = [
        [f"TP\n{metrics.true_positive}", f"FN\n{metrics.false_negative}"],
        [f"FP\n{metrics.false_positive}", f"TN\n{metrics.true_negative}"],
    ]

    fig, axis = plt.subplots(figsize=(7.2, 5.8))
    sns.heatmap(
        matrix,
        annot=annotations,
        fmt="",
        cmap="Blues",
        cbar=True,
        linewidths=2,
        linecolor="white",
        square=True,
        xticklabels=["预测：好瓜", "预测：坏瓜"],
        yticklabels=["真实：好瓜", "真实：坏瓜"],
        annot_kws={"fontsize": 17, "fontweight": "bold"},
        ax=axis,
    )
    axis.set_title("西瓜数据集 3.0a 混淆矩阵", fontsize=18, fontweight="bold", pad=18)
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.tick_params(axis="both", labelsize=12)

    metric_text = (
        f"Accuracy={metrics.accuracy:.4f}    "
        f"Precision={metrics.precision:.4f}    "
        f"Recall={metrics.recall:.4f}    "
        f"F1={metrics.f1:.4f}"
    )
    fig.text(0.5, 0.025, metric_text, ha="center", va="center", fontsize=12)
    save_figure(fig, output_path)


def plot_pr_curve(points: list[tuple[float, float]], metrics: Metrics, output_path: Path) -> None:
    recalls = [point[0] for point in points]
    precisions = [point[1] for point in points]

    fig, axis = plt.subplots(figsize=(8.2, 6.0))
    axis.step(recalls, precisions, where="post", color="#2563eb", linewidth=2.8, label=f"AUPRC = {metrics.auprc:.4f}")
    axis.scatter(recalls, precisions, color="#1d4ed8", edgecolor="white", s=46, zorder=3)
    axis.fill_between(recalls, precisions, step="post", alpha=0.18, color="#60a5fa")
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel("Recall", fontsize=13, fontweight="bold")
    axis.set_ylabel("Precision", fontsize=13, fontweight="bold")
    axis.set_title("西瓜数据集 3.0a PR 曲线", fontsize=18, fontweight="bold", pad=16)
    axis.legend(loc="lower left", frameon=True, fontsize=12)
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.text(
        0.98,
        0.93,
        f"Accuracy={metrics.accuracy:.4f}\nPrecision={metrics.precision:.4f}\nRecall={metrics.recall:.4f}\nF1={metrics.f1:.4f}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#bfdbfe", "alpha": 0.92},
    )
    save_figure(fig, output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="使用 matplotlib/seaborn 绘制混淆矩阵和 PR 曲线。")
    parser.add_argument("--input", type=Path, default=Path("results/训练集回代/watermelon_3a_predictions.csv"), help="预测结果 CSV 路径。")
    parser.add_argument("--confusion-output", type=Path, default=Path("results/训练集回代/watermelon_3a_confusion_matrix_mpl.svg"), help="混淆矩阵输出路径，支持 svg/png/pdf。")
    parser.add_argument("--pr-output", type=Path, default=Path("results/训练集回代/watermelon_3a_pr_curve_mpl.svg"), help="PR 曲线输出路径，支持 svg/png/pdf。")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_plot_style()
    rows = load_predictions(args.input)
    points, auprc = precision_recall_curve(rows)
    metrics = calculate_metrics(rows, auprc)
    plot_confusion_matrix(metrics, args.confusion_output)
    plot_pr_curve(points, metrics, args.pr_output)

    print("=== matplotlib/seaborn 分类结果图已生成 ===")
    print(f"输入预测文件: {args.input}")
    print(f"混淆矩阵: {args.confusion_output}")
    print(f"PR 曲线: {args.pr_output}")
    print(f"AUPRC: {metrics.auprc:.4f}")


if __name__ == "__main__":
    main()
