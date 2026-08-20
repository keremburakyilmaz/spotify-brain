from typing import Dict, Iterable

import numpy as np
import pandas as pd


AUDIO_FEATURES = [
    "valence",
    "energy",
    "danceability",
    "acousticness",
    "instrumentalness",
    "tempo_norm",
]


def cyclical_encode(value: float, max_value: float) -> tuple:
    return (
        np.sin(2 * np.pi * value / max_value),
        np.cos(2 * np.pi * value / max_value),
    )


def build_mood_features(
    window_df: pd.DataFrame,
    *,
    session_position: int,
    session_start_time,
) -> Dict[str, float]:
    """Build next-track features using only information available at prediction time."""
    if window_df.empty:
        raise ValueError("Mood prediction window cannot be empty")

    features: Dict[str, float] = {}
    for position, (_, row) in enumerate(window_df.iterrows()):
        features[f"mood_cluster_{position}"] = int(row["mood_cluster_id"])

    for feature in AUDIO_FEATURES:
        values = window_df[feature].astype(float).values
        features[f"{feature}_mean"] = float(np.mean(values))
        features[f"{feature}_std"] = float(np.std(values))

    last_track_time = pd.to_datetime(window_df.iloc[-1]["played_at"])
    hour_sin, hour_cos = cyclical_encode(last_track_time.hour, 24)
    day_sin, day_cos = cyclical_encode(last_track_time.dayofweek, 7)
    features.update(
        {
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "day_sin": day_sin,
            "day_cos": day_cos,
            "is_weekend": int(last_track_time.weekday() >= 5),
            "session_position": int(session_position),
            "time_since_session_start": float(
                (last_track_time - pd.to_datetime(session_start_time)).total_seconds() / 60
            ),
        }
    )

    for feature in AUDIO_FEATURES:
        features[f"current_{feature}"] = float(window_df.iloc[-1][feature])

    return features


def build_feature_vector(features: Dict[str, float], feature_cols: Iterable[str]) -> np.ndarray:
    """Order features for a saved model, including compatibility with older models."""
    values = []
    for column in feature_cols:
        if column == "session_length" and column not in features:
            # Older deployed models expect this leaked feature. Current position is the
            # only value available at prediction time and keeps updates deploy-safe until
            # the next full retrain removes the column from the model contract.
            values.append(features.get("session_position", 0.0))
        else:
            values.append(features.get(column, 0.0))
    return np.array([values])
