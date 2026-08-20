from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def recency_weights(timestamps: Iterable, half_life_days: float = 90.0) -> np.ndarray:
    values = pd.to_datetime(pd.Series(timestamps), utc=True)
    ages = (values.max() - values).dt.total_seconds() / 86400.0
    return np.power(0.5, ages.to_numpy() / max(float(half_life_days), 1.0))


def _sequence_columns(columns: Iterable[str]) -> List[str]:
    return sorted(
        (column for column in columns if column.startswith("mood_cluster_")),
        key=lambda column: int(column.rsplit("_", 1)[1]),
    )


def _context_keys(row, sequence_columns: List[str]) -> List[str]:
    sequence = [int(row[column]) for column in sequence_columns]
    angle = np.arctan2(float(row.get("hour_sin", 0.0)), float(row.get("hour_cos", 1.0)))
    hour = (angle % (2 * np.pi)) * 24 / (2 * np.pi)
    hour_bucket = int(hour // 4)
    position = int(row.get("session_position", len(sequence)))
    position_bucket = "early" if position <= 3 else "middle" if position <= 8 else "late"

    keys = []
    if len(sequence) >= 3:
        keys.append(f"seq3:{','.join(map(str, sequence[-3:]))}|h:{hour_bucket}|p:{position_bucket}")
    if len(sequence) >= 2:
        keys.append(f"seq2:{','.join(map(str, sequence[-2:]))}|h:{hour_bucket}")
    if sequence:
        keys.extend([
            f"last:{sequence[-1]}|h:{hour_bucket}",
            f"last:{sequence[-1]}",
        ])
    keys.append("global")
    return keys


def fit_contextual_transition(frame: pd.DataFrame, target_col: str,
                              classes: np.ndarray, sample_weight: np.ndarray,
                              smoothing: float = 2.0,
                              min_context_weight: float = 2.0) -> Dict:
    sequence_columns = _sequence_columns(frame.columns)
    counts: Dict[str, np.ndarray] = {}
    class_indices = {int(value): index for index, value in enumerate(classes)}
    for (_, row), target, weight in zip(
        frame.iterrows(), frame[target_col].to_numpy(), sample_weight
    ):
        for key in _context_keys(row, sequence_columns):
            counts.setdefault(key, np.zeros(len(classes), dtype=float))
            counts[key][class_indices[int(target)]] += float(weight)

    global_counts = counts.get("global", np.ones(len(classes), dtype=float))
    global_priors = global_counts / global_counts.sum()
    return {
        "classes": [int(value) for value in classes],
        "sequence_columns": sequence_columns,
        "counts": {key: values.tolist() for key, values in counts.items()},
        "global_priors": global_priors.tolist(),
        "smoothing": float(smoothing),
        "min_context_weight": float(min_context_weight),
    }


def predict_contextual_transition(frame: pd.DataFrame, artifact: Dict) -> Tuple[np.ndarray, List[str]]:
    counts = artifact["counts"]
    priors = np.asarray(artifact["global_priors"], dtype=float)
    smoothing = float(artifact.get("smoothing", 2.0))
    minimum = float(artifact.get("min_context_weight", 2.0))
    sequence_columns = artifact["sequence_columns"]
    probabilities = []
    levels = []

    for _, row in frame.iterrows():
        selected_key = "global"
        selected_counts = np.asarray(counts["global"], dtype=float)
        for key in _context_keys(row, sequence_columns):
            candidate = counts.get(key)
            if candidate is not None and (key == "global" or sum(candidate) >= minimum):
                selected_key = key
                selected_counts = np.asarray(candidate, dtype=float)
                break
        smoothed = selected_counts + smoothing * priors
        probabilities.append(smoothed / smoothed.sum())
        levels.append(selected_key.split(":", 1)[0])
    return np.asarray(probabilities), levels


def persistence_probabilities(frame: pd.DataFrame, classes: np.ndarray,
                              priors: np.ndarray, strength: float = 0.85) -> np.ndarray:
    sequence_columns = _sequence_columns(frame.columns)
    if not sequence_columns:
        return np.tile(priors, (len(frame), 1))
    output = np.tile((1.0 - strength) * priors, (len(frame), 1))
    class_indices = {int(value): index for index, value in enumerate(classes)}
    for row_index, value in enumerate(frame[sequence_columns[-1]].to_numpy()):
        if int(value) in class_indices:
            output[row_index, class_indices[int(value)]] += strength
    return output / output.sum(axis=1, keepdims=True)
