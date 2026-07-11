from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


MISSING_VALUES = {"", "?", None}


@dataclass
class C45Node:
    prediction: Any
    samples: int
    class_counts: dict[Any, int]
    feature_index: int | None = None
    feature_name: str | None = None
    threshold: float | None = None
    children: dict[Any, "C45Node"] = field(default_factory=dict)
    left: "C45Node | None" = None
    right: "C45Node | None" = None
    gain_ratio: float = 0.0
    is_leaf: bool = True


class C45DecisionTree:
    """A compact C4.5-style decision tree classifier.

    Implemented features:
    - information gain ratio for feature selection
    - binary threshold splits for continuous features
    - multiway splits for discrete features
    - recursive tree construction
    - prediction with majority-class fallback for missing or unseen values

    Missing-value weighting and pessimistic error pruning are intentionally
    simplified because the selected UCI datasets used in this lab have no
    missing values. Tree size is controlled by pre-pruning parameters.
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_gain_ratio: float = 1e-12,
    ) -> None:
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")
        if min_gain_ratio < 0:
            raise ValueError("min_gain_ratio must be non-negative.")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain_ratio = min_gain_ratio
        self.root: C45Node | None = None
        self.feature_names: list[str] = []
        self.feature_types: list[str] = []

    def fit(
        self,
        features: list[list[Any]],
        labels: list[Any],
        feature_names: list[str] | None = None,
        feature_types: list[str] | None = None,
    ) -> "C45DecisionTree":
        if not features:
            raise ValueError("No training data was provided.")
        if len(features) != len(labels):
            raise ValueError("features and labels must have the same length.")
        feature_count = len(features[0])
        if any(len(row) != feature_count for row in features):
            raise ValueError("All feature rows must have the same length.")

        self.feature_names = feature_names or [f"x{index}" for index in range(feature_count)]
        if len(self.feature_names) != feature_count:
            raise ValueError("feature_names length does not match feature count.")
        self.feature_types = feature_types or infer_feature_types(features)
        if len(self.feature_types) != feature_count:
            raise ValueError("feature_types length does not match feature count.")
        invalid_types = set(self.feature_types) - {"continuous", "discrete"}
        if invalid_types:
            raise ValueError(f"Unsupported feature types: {', '.join(sorted(invalid_types))}")

        available_features = list(range(feature_count))
        self.root = self._build(features, labels, available_features, depth=0)
        return self

    def predict_one(self, row: list[Any]) -> Any:
        if self.root is None:
            raise ValueError("Model has not been fitted yet.")
        node = self.root
        while not node.is_leaf:
            assert node.feature_index is not None
            value = row[node.feature_index]
            if is_missing(value):
                return node.prediction
            if node.threshold is not None:
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    return node.prediction
                next_node = node.left if numeric_value <= node.threshold else node.right
                if next_node is None:
                    return node.prediction
                node = next_node
            else:
                next_node = node.children.get(value)
                if next_node is None:
                    return node.prediction
                node = next_node
        return node.prediction

    def predict(self, features: list[list[Any]]) -> list[Any]:
        return [self.predict_one(row) for row in features]

    def _build(
        self,
        features: list[list[Any]],
        labels: list[Any],
        available_features: list[int],
        depth: int,
    ) -> C45Node:
        majority = majority_label(labels)
        node = C45Node(
            prediction=majority,
            samples=len(labels),
            class_counts=dict(Counter(labels)),
        )

        if len(set(labels)) == 1:
            return node
        if not available_features:
            return node
        if len(labels) < self.min_samples_split:
            return node
        if self.max_depth is not None and depth >= self.max_depth:
            return node

        best_split = self._best_split(features, labels, available_features)
        if best_split is None or best_split["gain_ratio"] < self.min_gain_ratio:
            return node

        feature_index = int(best_split["feature_index"])
        node.feature_index = feature_index
        node.feature_name = self.feature_names[feature_index]
        node.threshold = best_split.get("threshold")
        node.gain_ratio = float(best_split["gain_ratio"])
        node.is_leaf = False

        if node.threshold is not None:
            left_features, left_labels, right_features, right_labels = partition_continuous(
                features,
                labels,
                feature_index,
                node.threshold,
            )
            if not left_labels or not right_labels:
                node.is_leaf = True
                return node
            node.left = self._build(left_features, left_labels, available_features, depth + 1)
            node.right = self._build(right_features, right_labels, available_features, depth + 1)
        else:
            partitions = partition_discrete(features, labels, feature_index)
            if len(partitions) <= 1:
                node.is_leaf = True
                return node
            next_features = [index for index in available_features if index != feature_index]
            for feature_value, (child_features, child_labels) in partitions.items():
                node.children[feature_value] = self._build(child_features, child_labels, next_features, depth + 1)

        return node

    def _best_split(
        self,
        features: list[list[Any]],
        labels: list[Any],
        available_features: list[int],
    ) -> dict[str, Any] | None:
        best: dict[str, Any] | None = None
        for feature_index in available_features:
            if self.feature_types[feature_index] == "continuous":
                candidate = best_continuous_split(features, labels, feature_index)
            else:
                candidate = discrete_split_score(features, labels, feature_index)
            if candidate is None:
                continue
            candidate["feature_index"] = feature_index
            if best is None or candidate["gain_ratio"] > best["gain_ratio"]:
                best = candidate
        return best


def is_missing(value: Any) -> bool:
    return value in MISSING_VALUES


def infer_feature_types(features: list[list[Any]]) -> list[str]:
    feature_types: list[str] = []
    for column_index in range(len(features[0])):
        values = [row[column_index] for row in features if not is_missing(row[column_index])]
        try:
            for value in values:
                float(value)
        except (TypeError, ValueError):
            feature_types.append("discrete")
        else:
            feature_types.append("continuous")
    return feature_types


def majority_label(labels: list[Any]) -> Any:
    counts = Counter(labels)
    return sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))[0][0]


def entropy(labels: list[Any]) -> float:
    if not labels:
        return 0.0
    counts = Counter(labels)
    total = len(labels)
    value = 0.0
    for count in counts.values():
        probability = count / total
        value -= probability * math.log2(probability)
    return value


def split_info(partition_sizes: list[int]) -> float:
    total = sum(partition_sizes)
    if total == 0:
        return 0.0
    value = 0.0
    for size in partition_sizes:
        if size == 0:
            continue
        probability = size / total
        value -= probability * math.log2(probability)
    return value


def gain_ratio(parent_labels: list[Any], child_label_groups: list[list[Any]]) -> tuple[float, float, float]:
    total = len(parent_labels)
    if total == 0:
        return 0.0, 0.0, 0.0
    parent_entropy = entropy(parent_labels)
    weighted_child_entropy = sum((len(group) / total) * entropy(group) for group in child_label_groups)
    gain = parent_entropy - weighted_child_entropy
    info = split_info([len(group) for group in child_label_groups])
    ratio = gain / info if info > 0 else 0.0
    return gain, info, ratio


def best_continuous_split(
    features: list[list[Any]],
    labels: list[Any],
    feature_index: int,
) -> dict[str, Any] | None:
    pairs: list[tuple[float, Any]] = []
    for row, label in zip(features, labels):
        value = row[feature_index]
        if is_missing(value):
            continue
        pairs.append((float(value), label))
    if len(pairs) < 2:
        return None

    pairs.sort(key=lambda item: item[0])
    candidates: list[float] = []
    for (left_value, left_label), (right_value, right_label) in zip(pairs, pairs[1:]):
        if left_value == right_value:
            continue
        if left_label == right_label:
            continue
        candidates.append((left_value + right_value) / 2.0)

    if not candidates:
        unique_values = sorted(set(value for value, _ in pairs))
        candidates = [(left + right) / 2.0 for left, right in zip(unique_values, unique_values[1:])]
    if not candidates:
        return None

    best: dict[str, Any] | None = None
    valid_labels = [label for _, label in pairs]
    for threshold in candidates:
        left_labels = [label for value, label in pairs if value <= threshold]
        right_labels = [label for value, label in pairs if value > threshold]
        if not left_labels or not right_labels:
            continue
        gain, info, ratio = gain_ratio(valid_labels, [left_labels, right_labels])
        candidate = {
            "threshold": threshold,
            "gain": gain,
            "split_info": info,
            "gain_ratio": ratio,
        }
        if best is None or candidate["gain_ratio"] > best["gain_ratio"]:
            best = candidate
    return best


def discrete_split_score(
    features: list[list[Any]],
    labels: list[Any],
    feature_index: int,
) -> dict[str, Any] | None:
    partitions: dict[Any, list[Any]] = {}
    for row, label in zip(features, labels):
        value = row[feature_index]
        if is_missing(value):
            continue
        partitions.setdefault(value, []).append(label)
    if len(partitions) <= 1:
        return None
    valid_labels = [label for row, label in zip(features, labels) if not is_missing(row[feature_index])]
    gain, info, ratio = gain_ratio(valid_labels, list(partitions.values()))
    return {
        "threshold": None,
        "gain": gain,
        "split_info": info,
        "gain_ratio": ratio,
    }


def partition_continuous(
    features: list[list[Any]],
    labels: list[Any],
    feature_index: int,
    threshold: float,
) -> tuple[list[list[Any]], list[Any], list[list[Any]], list[Any]]:
    left_features: list[list[Any]] = []
    left_labels: list[Any] = []
    right_features: list[list[Any]] = []
    right_labels: list[Any] = []

    for row, label in zip(features, labels):
        value = row[feature_index]
        if is_missing(value):
            continue
        if float(value) <= threshold:
            left_features.append(row)
            left_labels.append(label)
        else:
            right_features.append(row)
            right_labels.append(label)
    return left_features, left_labels, right_features, right_labels


def partition_discrete(
    features: list[list[Any]],
    labels: list[Any],
    feature_index: int,
) -> dict[Any, tuple[list[list[Any]], list[Any]]]:
    partitions: dict[Any, tuple[list[list[Any]], list[Any]]] = {}
    for row, label in zip(features, labels):
        value = row[feature_index]
        if is_missing(value):
            continue
        if value not in partitions:
            partitions[value] = ([], [])
        partitions[value][0].append(row)
        partitions[value][1].append(label)
    return partitions
