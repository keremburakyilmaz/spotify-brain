from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    values = np.clip(values, 1e-12, None)
    return values / values.sum(axis=1, keepdims=True)


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    values = normalize_probabilities(probabilities)
    temperature = max(float(temperature), 0.05)
    logits = np.log(values) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    return normalize_probabilities(np.exp(logits))


def fit_temperature(y_true: Iterable[int], probabilities: np.ndarray,
                    classes: np.ndarray) -> float:
    y_true = np.asarray(list(y_true))
    candidates = np.geomspace(0.35, 3.0, 31)
    losses = [
        log_loss(y_true, apply_temperature(probabilities, value), labels=classes)
        for value in candidates
    ]
    return float(candidates[int(np.argmin(losses))])


def probability_metrics(y_true: Iterable[int], probabilities: np.ndarray,
                        classes: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(list(y_true))
    probabilities = normalize_probabilities(probabilities)
    predicted_indices = probabilities.argmax(axis=1)
    predictions = classes[predicted_indices]
    truth_indices = np.array([np.where(classes == value)[0][0] for value in y_true])
    ranked_indices = np.argsort(-probabilities, axis=1)
    ranks = np.array([
        int(np.where(row == truth_index)[0][0]) + 1
        for row, truth_index in zip(ranked_indices, truth_indices)
    ])
    one_hot = np.eye(len(classes))[truth_indices]

    confidence = probabilities.max(axis=1)
    correct = predictions == y_true
    calibration_error = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        in_bin = (confidence >= lower) & (
            confidence <= upper if upper >= 1.0 else confidence < upper
        )
        if in_bin.any():
            calibration_error += float(in_bin.mean()) * abs(
                float(correct[in_bin].mean()) - float(confidence[in_bin].mean())
            )

    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "f1_macro": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, probabilities, labels=classes)),
        "brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "mean_reciprocal_rank": float(np.mean(1.0 / ranks)),
        "expected_calibration_error": float(calibration_error),
    }
    for k in range(1, min(3, len(classes)) + 1):
        metrics[f"top_{k}_accuracy"] = float(np.mean(ranks <= k))
    return metrics


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    probabilities = normalize_probabilities(probabilities)
    if probabilities.shape[1] <= 1:
        return np.zeros(len(probabilities))
    entropy = -np.sum(probabilities * np.log(probabilities), axis=1)
    return entropy / np.log(probabilities.shape[1])


def derive_abstention_policy(y_true: Iterable[int], probabilities: np.ndarray,
                             classes: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(list(y_true))
    probabilities = normalize_probabilities(probabilities)
    predictions = classes[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)
    entropy = normalized_entropy(probabilities)
    overall_accuracy = float(np.mean(predictions == y_true))
    target_accuracy = min(0.8, overall_accuracy + 0.1)

    best: Tuple[float, float, float] = (0.5, 0.85, 0.0)
    for min_confidence in np.linspace(0.3, 0.75, 10):
        for max_entropy in np.linspace(0.55, 0.95, 9):
            accepted = (confidence >= min_confidence) & (entropy <= max_entropy)
            coverage = float(accepted.mean())
            if coverage < 0.2:
                continue
            selective_accuracy = float(np.mean(predictions[accepted] == y_true[accepted]))
            if selective_accuracy >= target_accuracy and coverage > best[2]:
                best = (float(min_confidence), float(max_entropy), coverage)

    min_confidence, max_entropy, coverage = best
    accepted = (confidence >= min_confidence) & (entropy <= max_entropy)
    coverage = float(accepted.mean())
    selective_accuracy = (
        float(np.mean(predictions[accepted] == y_true[accepted]))
        if accepted.any()
        else overall_accuracy
    )
    return {
        "min_confidence": min_confidence,
        "max_normalized_entropy": max_entropy,
        "validation_coverage": coverage,
        "validation_selective_accuracy": selective_accuracy,
    }


def describe_predictability(probabilities: np.ndarray,
                            policy: Dict[str, float]) -> Dict[str, float]:
    values = normalize_probabilities(np.asarray(probabilities, dtype=float).reshape(1, -1))
    confidence = float(values.max())
    entropy = float(normalized_entropy(values)[0])
    min_confidence = float(policy.get("min_confidence", 0.5))
    max_entropy = float(policy.get("max_normalized_entropy", 0.85))
    abstained = confidence < min_confidence or entropy > max_entropy
    if abstained:
        level = "low"
    elif confidence >= min(1.0, min_confidence + 0.2) and entropy <= max(0.0, max_entropy - 0.2):
        level = "high"
    else:
        level = "medium"
    return {
        "level": level,
        "normalized_entropy": entropy,
        "abstained": bool(abstained),
    }


def combine_switch_and_mood_probabilities(
    frame: pd.DataFrame,
    switch_probabilities: np.ndarray,
    mood_probabilities: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    """Compose P(switch) with conditional next-mood probabilities."""
    sequence_columns = sorted(
        (column for column in frame.columns if column.startswith("mood_cluster_")),
        key=lambda column: int(column.rsplit("_", 1)[1]),
    )
    if not sequence_columns:
        return normalize_probabilities(mood_probabilities)
    output = np.zeros_like(mood_probabilities, dtype=float)
    class_indices = {int(value): index for index, value in enumerate(classes)}
    previous_values = frame[sequence_columns[-1]].to_numpy(dtype=int)
    for row_index, previous in enumerate(previous_values):
        previous_index = class_indices[previous]
        switch_probability = float(np.clip(switch_probabilities[row_index], 0.0, 1.0))
        alternatives = np.asarray(mood_probabilities[row_index], dtype=float).copy()
        alternatives[previous_index] = 0.0
        if alternatives.sum() <= 0:
            alternatives[:] = 1.0
            alternatives[previous_index] = 0.0
        alternatives /= alternatives.sum()
        output[row_index] = alternatives * switch_probability
        output[row_index, previous_index] = 1.0 - switch_probability
    return normalize_probabilities(output)
