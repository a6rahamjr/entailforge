"""Dependency-light classification and calibration metrics."""

from typing import Dict, Iterable, List

import numpy as np


def confusion_matrix(
    labels: Iterable[int],
    predictions: Iterable[int],
) -> List[List[int]]:
    matrix = [[0, 0], [0, 0]]
    for label, prediction in zip(labels, predictions):
        matrix[int(label)][int(prediction)] += 1
    return matrix


def classification_metrics(
    labels: Iterable[int],
    predictions: Iterable[int],
) -> Dict[str, float | List[List[int]]]:
    labels_array = np.asarray(list(labels), dtype=np.int64)
    predictions_array = np.asarray(list(predictions), dtype=np.int64)
    if labels_array.size == 0:
        raise ValueError("Metrics require at least one prediction.")
    if labels_array.shape != predictions_array.shape:
        raise ValueError("Labels and predictions must have matching shapes.")

    matrix = confusion_matrix(labels_array, predictions_array)
    per_class = []
    for class_id in (0, 1):
        true_positive = matrix[class_id][class_id]
        false_positive = sum(matrix[row][class_id] for row in (0, 1)) - true_positive
        false_negative = sum(matrix[class_id]) - true_positive
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        per_class.append((precision, recall, f1))

    return {
        "accuracy": float(np.mean(labels_array == predictions_array)),
        "precision": float(np.mean([item[0] for item in per_class])),
        "recall": float(np.mean([item[1] for item in per_class])),
        "macro_f1": float(np.mean([item[2] for item in per_class])),
        "confusion_matrix": matrix,
    }


def expected_calibration_error(
    labels: Iterable[int],
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    labels_array = np.asarray(list(labels), dtype=np.int64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("Probabilities must have shape [n, 2].")
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0

    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        if index == 0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences > lower) & (confidences <= upper)
        if not mask.any():
            continue
        bin_accuracy = np.mean(predictions[mask] == labels_array[mask])
        bin_confidence = np.mean(confidences[mask])
        error += np.mean(mask) * abs(bin_accuracy - bin_confidence)

    return float(error)
