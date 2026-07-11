from __future__ import annotations

import csv
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from c45_decision_tree import C45DecisionTree


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
LIBSVM_DIR = ROOT / "libsvm-3.31" / "windows"
SVM_SCALE = LIBSVM_DIR / "svm-scale.exe"
SVM_TRAIN = LIBSVM_DIR / "svm-train.exe"
SVM_PREDICT = LIBSVM_DIR / "svm-predict.exe"


def configure_plot_style() -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]:
        if font_name in available_fonts:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [font_name]
            break
    plt.rcParams["axes.unicode_minus"] = False


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: list[list[int]]


@dataclass
class RegressionMetrics:
    mse: float
    rmse: float
    mae: float
    squared_correlation: float


def ensure_libsvm() -> None:
    missing = [path for path in [SVM_SCALE, SVM_TRAIN, SVM_PREDICT] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing LIBSVM executables: " + ", ".join(str(path) for path in missing))


def run_command(args: list[str], stdout_path: Path | None = None) -> str:
    if stdout_path:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("w", encoding="utf-8") as output_file:
            completed = subprocess.run(args, cwd=ROOT, text=True, stdout=output_file, stderr=subprocess.PIPE, check=True)
        return completed.stderr
    completed = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=True)
    return completed.stdout + completed.stderr


def read_libsvm_labels(path: Path) -> list[float]:
    labels: list[float] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                labels.append(float(line.split()[0]))
    return labels


def read_prediction_labels(path: Path) -> list[float]:
    return [float(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def binary_metrics(y_true: list[float], y_pred: list[float], positive_label: float = 1.0) -> ClassificationMetrics:
    true_binary = [1 if value == positive_label else 0 for value in y_true]
    pred_binary = [1 if value == positive_label else 0 for value in y_pred]
    matrix = confusion_matrix(true_binary, pred_binary, labels=[1, 0])
    return ClassificationMetrics(
        accuracy=accuracy_score(true_binary, pred_binary),
        precision=precision_score(true_binary, pred_binary, zero_division=0),
        recall=recall_score(true_binary, pred_binary, zero_division=0),
        f1=f1_score(true_binary, pred_binary, zero_division=0),
        confusion=matrix.tolist(),
    )


def multiclass_metrics(y_true: list[Any], y_pred: list[Any]) -> ClassificationMetrics:
    labels = sorted(set(y_true) | set(y_pred), key=str)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return ClassificationMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, average="macro", zero_division=0),
        recall=recall_score(y_true, y_pred, average="macro", zero_division=0),
        f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
        confusion=matrix.tolist(),
    )


def save_confusion_matrix(matrix: list[list[int]], labels: list[str], title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(5.8, 4.8))
    array = np.array(matrix)
    image = axis.imshow(array, cmap="Blues")
    axis.set_xticks(range(len(labels)), labels=[f"Pred {label}" for label in labels])
    axis.set_yticks(range(len(labels)), labels=[f"True {label}" for label in labels])
    axis.set_title(title)
    for row in range(array.shape[0]):
        for col in range(array.shape[1]):
            axis.text(col, row, str(array[row, col]), ha="center", va="center", fontsize=12)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def save_metrics_bar(rows: list[dict[str, Any]], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(rows))
    width = 0.18
    fig, axis = plt.subplots(figsize=(8.6, 5.2))
    for offset, metric in enumerate(metric_names):
        axis.bar(x + (offset - 1.5) * width, [row[metric] for row in rows], width, label=metric)
    axis.set_xticks(x, [row["model"] for row in rows], rotation=18, ha="right")
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title(title)
    axis.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def write_libsvm_file(path: Path, labels: list[float], features: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        for label, row in zip(labels, features):
            parts = [format_label(label)]
            parts.extend(f"{index + 1}:{value:.10g}" for index, value in enumerate(row))
            file.write(" ".join(parts) + "\n")


def format_label(label: float) -> str:
    return str(int(label)) if float(label).is_integer() else f"{label:.10g}"


def parse_model_metadata(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {"sv_lines": []}
    in_sv = False
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line == "SV":
            in_sv = True
            continue
        if in_sv:
            metadata["sv_lines"].append(line)
            continue
        if line.startswith("total_sv "):
            metadata["total_sv"] = int(line.split()[1])
        elif line.startswith("nr_sv "):
            metadata["nr_sv"] = [int(value) for value in line.split()[1:]]
        elif line.startswith("rho "):
            metadata["rho"] = [float(value) for value in line.split()[1:]]
    return metadata


def parse_support_vector_feature(line: str) -> dict[int, float]:
    tokens = line.split()[1:]
    features: dict[int, float] = {}
    for token in tokens:
        if ":" not in token:
            continue
        index, value = token.split(":", 1)
        features[int(index)] = float(value)
    return features


def read_watermelon_svm() -> tuple[list[int], np.ndarray, list[int], list[str]]:
    row_ids: list[int] = []
    labels: list[int] = []
    features: list[list[float]] = []
    for index, line in enumerate((DATA_DIR / "watermelon_3a_svm.txt").read_text(encoding="utf-8").splitlines(), start=1):
        tokens = line.split()
        labels.append(int(float(tokens[0])))
        row_ids.append(index)
        feature_values = [0.0, 0.0]
        for token in tokens[1:]:
            feature_index, value = token.split(":")
            feature_values[int(feature_index) - 1] = float(value)
        features.append(feature_values)
    return row_ids, np.array(features), labels, ["密度", "含糖率"]


def run_6_2() -> list[dict[str, Any]]:
    output_dir = RESULTS_DIR / "6.2"
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    scaled_path = output_dir / "watermelon_3a_svm_scaled.txt"
    run_command([str(SVM_SCALE), "-l", "0", "-u", "1", str(DATA_DIR / "watermelon_3a_svm.txt")], stdout_path=scaled_path)

    configs = [
        {"model": "linear", "kernel": "linear", "args": ["-s", "0", "-t", "0", "-c", "1"]},
        {"model": "rbf", "kernel": "RBF", "args": ["-s", "0", "-t", "2", "-c", "1", "-g", "1"]},
    ]
    y_true = read_libsvm_labels(scaled_path)
    row_ids, original_features, original_labels, _ = read_watermelon_svm()
    scaled_features = load_libsvm_features(scaled_path, feature_count=2)
    min_values = original_features.min(axis=0)
    max_values = original_features.max(axis=0)

    summary_rows: list[dict[str, Any]] = []
    prediction_columns: dict[str, list[float]] = {}
    for config in configs:
        model_path = output_dir / f"watermelon_{config['model']}.model"
        prediction_path = output_dir / f"watermelon_{config['model']}_predict.txt"
        train_output = run_command([str(SVM_TRAIN), *config["args"], str(scaled_path), str(model_path)])
        predict_output = run_command([str(SVM_PREDICT), str(scaled_path), str(model_path), str(prediction_path)])
        y_pred = read_prediction_labels(prediction_path)
        metrics = binary_metrics(y_true, y_pred)
        metadata = parse_model_metadata(model_path)
        support_ids = match_support_vectors(metadata["sv_lines"], scaled_features, row_ids)
        prediction_columns[config["model"]] = y_pred
        summary_rows.append(
            {
                "model": config["model"],
                "kernel": config["kernel"],
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "tp": metrics.confusion[0][0],
                "fn": metrics.confusion[0][1],
                "fp": metrics.confusion[1][0],
                "tn": metrics.confusion[1][1],
                "total_sv": metadata.get("total_sv", 0),
                "nr_sv": " ".join(str(value) for value in metadata.get("nr_sv", [])),
                "support_ids": " ".join(str(value) for value in support_ids),
                "train_output": train_output.strip(),
                "predict_output": predict_output.strip(),
            }
        )
        save_confusion_matrix(metrics.confusion, ["好瓜", "坏瓜"], f"6.2 {config['kernel']} SVM Confusion Matrix", figure_dir / f"watermelon_{config['model']}_confusion_matrix.svg")

    write_6_2_predictions(output_dir / "watermelon_svm_predictions.csv", row_ids, original_features, original_labels, prediction_columns)
    write_csv(output_dir / "watermelon_svm_metrics.csv", summary_rows)
    save_metrics_bar(summary_rows, figure_dir / "watermelon_svm_metrics_bar.svg", "6.2 SVM Metrics on Watermelon 3.0a")
    save_decision_boundary_6_2(configs, output_dir, figure_dir / "watermelon_svm_decision_boundary.svg", original_features, original_labels, min_values, max_values)
    write_6_2_summary(output_dir / "watermelon_svm_summary.md", summary_rows)
    return summary_rows


def load_libsvm_features(path: Path, feature_count: int) -> np.ndarray:
    features: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = [0.0] * feature_count
        for token in line.split()[1:]:
            index, value = token.split(":")
            row[int(index) - 1] = float(value)
        features.append(row)
    return np.array(features)


def match_support_vectors(sv_lines: list[str], scaled_features: np.ndarray, row_ids: list[int]) -> list[int]:
    support_ids: list[int] = []
    used: set[int] = set()
    for line in sv_lines:
        sv_features = parse_support_vector_feature(line)
        vector = np.array([sv_features.get(index + 1, 0.0) for index in range(scaled_features.shape[1])])
        distances = np.linalg.norm(scaled_features - vector, axis=1)
        order = np.argsort(distances)
        for candidate in order:
            if int(candidate) not in used and distances[candidate] < 1e-6:
                used.add(int(candidate))
                support_ids.append(row_ids[int(candidate)])
                break
    return support_ids


def write_6_2_predictions(path: Path, row_ids: list[int], features: np.ndarray, labels: list[int], prediction_columns: dict[str, list[float]]) -> None:
    rows = []
    for index, row_id in enumerate(row_ids):
        rows.append(
            {
                "编号": row_id,
                "密度": f"{features[index, 0]:.3f}",
                "含糖率": f"{features[index, 1]:.3f}",
                "真实标签": "是" if labels[index] == 1 else "否",
                "线性核预测": "是" if prediction_columns["linear"][index] == 1 else "否",
                "RBF核预测": "是" if prediction_columns["rbf"][index] == 1 else "否",
            }
        )
    write_csv(path, rows)


def save_decision_boundary_6_2(configs: list[dict[str, Any]], output_dir: Path, output_path: Path, features: np.ndarray, labels: list[int], min_values: np.ndarray, max_values: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_min, x_max = features[:, 0].min() - 0.03, features[:, 0].max() + 0.03
    y_min, y_max = features[:, 1].min() - 0.03, features[:, 1].max() + 0.03
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 180), np.linspace(y_min, y_max, 180))
    grid_original = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = (grid_original - min_values) / (max_values - min_values)
    grid_scaled = np.clip(grid_scaled, 0.0, 1.0)
    grid_path = output_dir / "watermelon_grid_scaled.txt"
    write_libsvm_file(grid_path, [0.0] * len(grid_scaled), grid_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharex=True, sharey=True)
    for axis, config in zip(axes, configs):
        model_path = output_dir / f"watermelon_{config['model']}.model"
        prediction_path = output_dir / f"watermelon_{config['model']}_grid_predict.txt"
        run_command([str(SVM_PREDICT), str(grid_path), str(model_path), str(prediction_path)])
        grid_pred = np.array(read_prediction_labels(prediction_path)).reshape(xx.shape)
        axis.contourf(xx, yy, grid_pred, levels=[-1.5, 0, 1.5], colors=["#fee2e2", "#dcfce7"], alpha=0.75)
        positive = np.array(labels) == 1
        axis.scatter(features[positive, 0], features[positive, 1], c="#15803d", edgecolor="white", label="好瓜", s=60)
        axis.scatter(features[~positive, 0], features[~positive, 1], c="#b91c1c", marker="s", edgecolor="white", label="坏瓜", s=60)
        metadata = parse_model_metadata(model_path)
        scaled_data = (features - min_values) / (max_values - min_values)
        support_ids = match_support_vectors(metadata["sv_lines"], scaled_data, list(range(1, len(labels) + 1)))
        support_indices = [sample_id - 1 for sample_id in support_ids]
        axis.scatter(features[support_indices, 0], features[support_indices, 1], facecolors="none", edgecolor="#111827", s=150, linewidths=1.8, label="支持向量")
        axis.set_title(f"{config['kernel']} SVM")
        axis.set_xlabel("密度")
        axis.grid(True, linestyle="--", alpha=0.25)
    axes[0].set_ylabel("含糖率")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def run_6_3() -> list[dict[str, Any]]:
    output_dir = RESULTS_DIR / "6.3"
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [load_wine_dataset(), load_wdbc_dataset()]
    all_metrics: list[dict[str, Any]] = []
    for dataset_name, features, labels, feature_names in datasets:
        dataset_slug = slugify(dataset_name)
        encoded_labels = LabelEncoder().fit_transform(labels)
        label_encoder = LabelEncoder()
        numeric_labels = label_encoder.fit_transform(labels)
        train_x, test_x, train_y, test_y, train_names, test_names = train_test_split(
            features,
            numeric_labels,
            labels,
            test_size=0.30,
            random_state=42,
            stratify=numeric_labels,
        )
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_x)
        test_scaled = scaler.transform(test_x)

        libsvm_labels_train = [float(value + 1) for value in train_y]
        libsvm_labels_test = [float(value + 1) for value in test_y]
        train_path = output_dir / f"{dataset_slug}_train.libsvm"
        test_path = output_dir / f"{dataset_slug}_test.libsvm"
        write_libsvm_file(train_path, libsvm_labels_train, train_scaled)
        write_libsvm_file(test_path, libsvm_labels_test, test_scaled)

        dataset_rows: list[dict[str, Any]] = []
        prediction_rows: list[dict[str, Any]] = []
        for model_name, svm_args in [
            ("SVM-linear", ["-s", "0", "-t", "0", "-c", "1"]),
            ("SVM-RBF", ["-s", "0", "-t", "2", "-c", "1", "-g", "0.5"]),
        ]:
            model_path = output_dir / f"{dataset_slug}_{model_name.lower().replace('-', '_')}.model"
            prediction_path = output_dir / f"{dataset_slug}_{model_name.lower().replace('-', '_')}_predict.txt"
            run_command([str(SVM_TRAIN), *svm_args, str(train_path), str(model_path)])
            run_command([str(SVM_PREDICT), str(test_path), str(model_path), str(prediction_path)])
            pred_numeric = [int(value) - 1 for value in read_prediction_labels(prediction_path)]
            pred_names = label_encoder.inverse_transform(pred_numeric)
            metrics = multiclass_metrics(list(test_names), list(pred_names))
            row = metric_row(dataset_name, model_name, metrics, "LIBSVM " + " ".join(svm_args))
            dataset_rows.append(row)
            all_metrics.append(row)
            append_prediction_rows(prediction_rows, dataset_name, model_name, test_names, pred_names)
            save_confusion_matrix(metrics.confusion, [str(label) for label in label_encoder.classes_], f"{dataset_name} {model_name}", output_dir / "confusion_matrices" / f"{dataset_slug}_{slugify(model_name)}_confusion_matrix.svg")

        bp_model = MLPClassifier(hidden_layer_sizes=(16,), activation="relu", solver="adam", max_iter=2000, random_state=42)
        bp_model.fit(train_scaled, train_y)
        bp_pred = bp_model.predict(test_scaled)
        bp_names = label_encoder.inverse_transform(bp_pred)
        bp_metrics = multiclass_metrics(list(test_names), list(bp_names))
        bp_row = metric_row(dataset_name, "BP", bp_metrics, "MLP hidden=(16,), relu, adam")
        dataset_rows.append(bp_row)
        all_metrics.append(bp_row)
        append_prediction_rows(prediction_rows, dataset_name, "BP", test_names, bp_names)
        save_confusion_matrix(bp_metrics.confusion, [str(label) for label in label_encoder.classes_], f"{dataset_name} BP", output_dir / "confusion_matrices" / f"{dataset_slug}_bp_confusion_matrix.svg")

        c45_model = C45DecisionTree(max_depth=8, min_samples_split=3, min_gain_ratio=1e-8)
        c45_model.fit(train_scaled.tolist(), list(train_names), feature_names=feature_names, feature_types=["continuous"] * train_scaled.shape[1])
        c45_pred = c45_model.predict(test_scaled.tolist())
        c45_metrics = multiclass_metrics(list(test_names), list(c45_pred))
        c45_row = metric_row(dataset_name, "C4.5", c45_metrics, "handwritten gain-ratio tree")
        dataset_rows.append(c45_row)
        all_metrics.append(c45_row)
        append_prediction_rows(prediction_rows, dataset_name, "C4.5", test_names, c45_pred)
        save_confusion_matrix(c45_metrics.confusion, [str(label) for label in label_encoder.classes_], f"{dataset_name} C4.5", output_dir / "confusion_matrices" / f"{dataset_slug}_c45_confusion_matrix.svg")

        write_csv(output_dir / f"{dataset_slug}_predictions.csv", prediction_rows)
        save_metrics_bar(dataset_rows, output_dir / "figures" / f"{dataset_slug}_metrics_bar.svg", f"6.3 {dataset_name} Model Comparison")

    write_csv(output_dir / "uci_metrics.csv", all_metrics)
    write_6_3_summary(output_dir / "uci_comparison_summary.md", all_metrics)
    return all_metrics


def load_wine_dataset() -> tuple[str, np.ndarray, np.ndarray, list[str]]:
    rows = list(csv.reader((DATA_DIR / "uci" / "Wine" / "wine.data").open(encoding="utf-8")))
    labels = np.array([row[0] for row in rows])
    features = np.array([[float(value) for value in row[1:]] for row in rows])
    names = [
        "Alcohol",
        "Malic acid",
        "Ash",
        "Alcalinity of ash",
        "Magnesium",
        "Total phenols",
        "Flavanoids",
        "Nonflavanoid phenols",
        "Proanthocyanins",
        "Color intensity",
        "Hue",
        "OD280/OD315",
        "Proline",
    ]
    return "Wine", features, labels, names


def load_wdbc_dataset() -> tuple[str, np.ndarray, np.ndarray, list[str]]:
    rows = list(csv.reader((DATA_DIR / "uci" / "breast_cancer" / "wdbc.data").open(encoding="utf-8")))
    labels = np.array([row[1] for row in rows])
    features = np.array([[float(value) for value in row[2:]] for row in rows])
    names = [f"x{index}" for index in range(features.shape[1])]
    return "Breast Cancer WDBC", features, labels, names


def append_prediction_rows(rows: list[dict[str, Any]], dataset: str, model: str, y_true: list[Any], y_pred: list[Any]) -> None:
    for index, (true_label, pred_label) in enumerate(zip(y_true, y_pred), start=1):
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "test_index": index,
                "true_label": true_label,
                "predicted_label": pred_label,
                "correct": true_label == pred_label,
            }
        )


def metric_row(dataset: str, model: str, metrics: ClassificationMetrics, params: str) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "model": model,
        "params": params,
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
    }


def run_6_8() -> list[dict[str, Any]]:
    output_dir = RESULTS_DIR / "6.8"
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    scaled_path = output_dir / "watermelon_3a_svr_scaled.txt"
    run_command([str(SVM_SCALE), "-l", "0", "-u", "1", str(DATA_DIR / "watermelon_3a_svr.txt")], stdout_path=scaled_path)
    density, sugar = read_watermelon_svr_original()
    min_density = float(density.min())
    max_density = float(density.max())

    configs = [
        {"model": "linear", "kernel": "linear", "args": ["-s", "3", "-t", "0", "-c", "1", "-p", "0.05"]},
        {"model": "rbf", "kernel": "RBF", "args": ["-s", "3", "-t", "2", "-c", "1", "-g", "1", "-p", "0.05"]},
    ]
    predictions: dict[str, list[float]] = {}
    metric_rows: list[dict[str, Any]] = []
    for config in configs:
        model_path = output_dir / f"watermelon_svr_{config['model']}.model"
        prediction_path = output_dir / f"watermelon_svr_{config['model']}_predict.txt"
        run_command([str(SVM_TRAIN), *config["args"], str(scaled_path), str(model_path)])
        output = run_command([str(SVM_PREDICT), str(scaled_path), str(model_path), str(prediction_path)])
        y_pred = read_prediction_labels(prediction_path)
        predictions[config["model"]] = y_pred
        metrics = regression_metrics(list(sugar), y_pred)
        metadata = parse_model_metadata(model_path)
        metric_rows.append(
            {
                "model": config["model"],
                "kernel": config["kernel"],
                "params": " ".join(config["args"]),
                "mse": metrics.mse,
                "rmse": metrics.rmse,
                "mae": metrics.mae,
                "squared_correlation": metrics.squared_correlation,
                "total_sv": metadata.get("total_sv", 0),
                "libsvm_output": output.strip(),
            }
        )

    write_6_8_predictions(output_dir / "watermelon_svr_predictions.csv", density, sugar, predictions)
    write_csv(output_dir / "watermelon_svr_metrics.csv", metric_rows)
    save_svr_fit_curve(configs, output_dir, figure_dir / "watermelon_svr_fit_curve.svg", density, sugar, min_density, max_density)
    save_true_vs_predicted(figure_dir / "watermelon_svr_true_vs_predicted.svg", sugar, predictions)
    save_error_bar(figure_dir / "watermelon_svr_error_bar.svg", sugar, predictions)
    write_6_8_summary(output_dir / "watermelon_svr_summary.md", metric_rows)
    return metric_rows


def read_watermelon_svr_original() -> tuple[np.ndarray, np.ndarray]:
    densities: list[float] = []
    sugars: list[float] = []
    for line in (DATA_DIR / "watermelon_3a_svr.txt").read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        sugars.append(float(tokens[0]))
        densities.append(float(tokens[1].split(":")[1]))
    return np.array(densities), np.array(sugars)


def regression_metrics(y_true: list[float], y_pred: list[float]) -> RegressionMetrics:
    true_array = np.array(y_true, dtype=float)
    pred_array = np.array(y_pred, dtype=float)
    errors = pred_array - true_array
    mse = float(np.mean(errors * errors))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(errors)))
    corr = np.corrcoef(true_array, pred_array)[0, 1]
    squared_correlation = float(corr * corr) if not np.isnan(corr) else 0.0
    return RegressionMetrics(mse, rmse, mae, squared_correlation)


def write_6_8_predictions(path: Path, density: np.ndarray, sugar: np.ndarray, predictions: dict[str, list[float]]) -> None:
    rows = []
    for index in range(len(sugar)):
        linear_pred = predictions["linear"][index]
        rbf_pred = predictions["rbf"][index]
        rows.append(
            {
                "编号": index + 1,
                "密度": f"{density[index]:.3f}",
                "真实含糖率": f"{sugar[index]:.3f}",
                "线性SVR预测": f"{linear_pred:.6f}",
                "RBF-SVR预测": f"{rbf_pred:.6f}",
                "线性SVR误差": f"{linear_pred - sugar[index]:.6f}",
                "RBF-SVR误差": f"{rbf_pred - sugar[index]:.6f}",
            }
        )
    write_csv(path, rows)


def save_svr_fit_curve(configs: list[dict[str, Any]], output_dir: Path, output_path: Path, density: np.ndarray, sugar: np.ndarray, min_density: float, max_density: float) -> None:
    grid_density = np.linspace(float(density.min()) - 0.03, float(density.max()) + 0.03, 200)
    grid_scaled = ((grid_density - min_density) / (max_density - min_density)).reshape(-1, 1)
    grid_scaled = np.clip(grid_scaled, 0.0, 1.0)
    grid_path = output_dir / "watermelon_svr_grid_scaled.txt"
    write_libsvm_file(grid_path, [0.0] * len(grid_scaled), grid_scaled)
    fig, axis = plt.subplots(figsize=(7.4, 5.0))
    axis.scatter(density, sugar, c="#111827", label="真实样本", s=48)
    for config in configs:
        model_path = output_dir / f"watermelon_svr_{config['model']}.model"
        prediction_path = output_dir / f"watermelon_svr_{config['model']}_grid_predict.txt"
        run_command([str(SVM_PREDICT), str(grid_path), str(model_path), str(prediction_path)])
        curve = read_prediction_labels(prediction_path)
        axis.plot(grid_density, curve, linewidth=2.2, label=f"{config['kernel']} SVR")
    axis.set_xlabel("密度")
    axis.set_ylabel("含糖率")
    axis.set_title("6.8 SVR Fit Curve")
    axis.legend()
    axis.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def save_true_vs_predicted(output_path: Path, y_true: np.ndarray, predictions: dict[str, list[float]]) -> None:
    fig, axis = plt.subplots(figsize=(6.2, 5.2))
    low = min(float(y_true.min()), min(min(values) for values in predictions.values())) - 0.02
    high = max(float(y_true.max()), max(max(values) for values in predictions.values())) + 0.02
    axis.plot([low, high], [low, high], color="#6b7280", linestyle="--", label="理想预测")
    for name, values in predictions.items():
        axis.scatter(y_true, values, label=name, s=48)
    axis.set_xlabel("真实含糖率")
    axis.set_ylabel("预测含糖率")
    axis.set_title("6.8 True vs Predicted Sugar Rate")
    axis.legend()
    axis.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def save_error_bar(output_path: Path, y_true: np.ndarray, predictions: dict[str, list[float]]) -> None:
    x = np.arange(1, len(y_true) + 1)
    fig, axis = plt.subplots(figsize=(9.2, 4.8))
    axis.bar(x - 0.18, np.array(predictions["linear"]) - y_true, width=0.36, label="linear SVR")
    axis.bar(x + 0.18, np.array(predictions["rbf"]) - y_true, width=0.36, label="RBF SVR")
    axis.axhline(0, color="#374151", linewidth=1)
    axis.set_xlabel("样本编号")
    axis.set_ylabel("预测误差")
    axis.set_title("6.8 SVR Prediction Error")
    axis.legend()
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_6_2_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# 6.2 西瓜 3.0a SVM 分类结果", "", "| 模型 | Accuracy | Precision | Recall | F1 | TP | TN | FP | FN | total_sv | nr_sv | 支持向量编号 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for row in rows:
        lines.append(
            f"| {row['kernel']} | {row['accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | "
            f"{row['tp']} | {row['tn']} | {row['fp']} | {row['fn']} | {row['total_sv']} | {row['nr_sv']} | {row['support_ids']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_6_3_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# 6.3 UCI 数据集分类比较结果", "", "| 数据集 | 模型 | Accuracy | Precision macro | Recall macro | F1 macro | 参数 |", "|---|---|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['params']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_6_8_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# 6.8 西瓜 3.0a SVR 回归结果", "", "| 模型 | MSE | RMSE | MAE | Squared correlation | total_sv | 参数 |", "|---|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(
            f"| {row['kernel']} SVR | {row['mse']:.6f} | {row['rmse']:.6f} | {row['mae']:.6f} | {row['squared_correlation']:.6f} | {row['total_sv']} | {row['params']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def slugify(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_").replace(".", "")


def main() -> None:
    configure_plot_style()
    ensure_libsvm()
    run_6_2()
    run_6_3()
    run_6_8()
    print("lab2 experiments completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
