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
    mood_sequence = []
    for position, (_, row) in enumerate(window_df.iterrows()):
        mood = int(row["mood_cluster_id"])
        mood_sequence.append(mood)
        features[f"mood_cluster_{position}"] = mood

    transitions = np.diff(mood_sequence) if len(mood_sequence) > 1 else np.array([])
    features.update(
        {
            "mood_unique_count": int(len(set(mood_sequence))),
            "mood_transition_count": int(np.count_nonzero(transitions)),
            "mood_repeat_ratio": float(
                1.0 - np.count_nonzero(transitions) / max(1, len(mood_sequence) - 1)
            ),
            "mood_dominant_share": float(
                pd.Series(mood_sequence).value_counts(normalize=True).iloc[0]
            ),
            "mood_changed_last": int(
                len(mood_sequence) > 1 and mood_sequence[-1] != mood_sequence[-2]
            ),
        }
    )

    for feature in AUDIO_FEATURES:
        values = window_df[feature].astype(float).values
        features[f"{feature}_mean"] = float(np.mean(values))
        features[f"{feature}_std"] = float(np.std(values))
        features[f"{feature}_trend"] = float(
            np.polyfit(np.arange(len(values)), values, 1)[0]
            if len(values) > 1
            else 0.0
        )
        features[f"{feature}_last_delta"] = float(
            values[-1] - values[-2] if len(values) > 1 else 0.0
        )

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

    timestamps = pd.to_datetime(window_df["played_at"])
    gaps = timestamps.diff().dt.total_seconds().dropna().to_numpy() / 60.0
    features.update(
        {
            "last_track_gap_minutes": float(gaps[-1]) if len(gaps) else 0.0,
            "mean_track_gap_minutes": float(np.mean(gaps)) if len(gaps) else 0.0,
            "std_track_gap_minutes": float(np.std(gaps)) if len(gaps) else 0.0,
            "artist_unique_ratio": float(
                window_df["artist_name"].nunique() / len(window_df)
                if "artist_name" in window_df
                else 0.0
            ),
            "track_repeat_ratio": float(
                1.0 - window_df["track_id"].nunique() / len(window_df)
                if "track_id" in window_df
                else 0.0
            ),
        }
    )

    # Richer behavioral exports can populate these columns in the future. The
    # availability flags prevent a missing signal from looking like a real zero.
    optional_binary_signals = {
        "skipped": "skip_rate",
        "shuffle_state": "shuffle_rate",
        "manually_selected": "manual_selection_rate",
    }
    for source_column, feature_name in optional_binary_signals.items():
        available = source_column in window_df and window_df[source_column].notna().any()
        features[f"{feature_name}_available"] = int(available)
        features[feature_name] = (
            float(window_df[source_column].dropna().astype(float).mean()) if available else 0.0
        )

    completion_available = (
        "ms_played" in window_df
        and "duration_ms" in window_df
        and (window_df["duration_ms"].fillna(0) > 0).any()
    )
    features["completion_rate_available"] = int(completion_available)
    if completion_available:
        completion = window_df["ms_played"] / window_df["duration_ms"].replace(0, np.nan)
        features["completion_rate_mean"] = float(completion.clip(0, 1).dropna().mean())
    else:
        features["completion_rate_mean"] = 0.0

    source_column = next(
        (column for column in ["playback_source", "context_uri"] if column in window_df),
        None,
    )
    source_available = bool(source_column and window_df[source_column].notna().any())
    features["playback_source_available"] = int(source_available)
    features["playback_source_continuity"] = (
        float(window_df[source_column].dropna().value_counts(normalize=True).iloc[0])
        if source_available
        else 0.0
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
