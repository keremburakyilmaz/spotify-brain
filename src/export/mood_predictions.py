import pandas as pd
import numpy as np
import json
import pickle
from typing import Dict


def cyclical_encode(value: float, max_value: float) -> tuple:
    sin_val = np.sin(2 * np.pi * value / max_value)
    cos_val = np.cos(2 * np.pi * value / max_value)
    return sin_val, cos_val


def load_model(model_path: str) -> tuple:
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data["model"], model_data["feature_cols"]


def build_next_track_prediction(history_path: str, mood_model_path: str, 
                                mood_clusters_path: str, window_size: int = 3) -> Dict:
    # Load history
    df = pd.read_parquet(history_path)
    df = df.dropna(subset=["mood_cluster_id", "session_id"])
    df = df.sort_values(["session_id", "played_at"]).reset_index(drop=True)
    
    # Get most recent session
    latest_session_id = df["session_id"].max()
    session_df = df[df["session_id"] == latest_session_id].copy()
    
    if len(session_df) < window_size:
        # Use last N tracks across sessions
        session_df = df.tail(window_size).copy()
    
    # Get last window_size tracks
    window_df = session_df.tail(window_size).copy()
    
    # Load model
    model, feature_cols = load_model(mood_model_path)
    
    # Build features (same as in build_mood_dataset)
    audio_features = ["valence", "energy", "danceability", "acousticness", 
                     "instrumentalness", "tempo_norm"]
    
    features = {}
    
    # Sequence features
    for j, (idx, row) in enumerate(window_df.iterrows()):
        features[f"mood_cluster_{j}"] = int(row["mood_cluster_id"])
    
    # Aggregated audio features
    for feat in audio_features:
        values = window_df[feat].dropna().values
        if len(values) > 0:
            features[f"{feat}_mean"] = float(np.mean(values))
            features[f"{feat}_std"] = float(np.std(values))
        else:
            features[f"{feat}_mean"] = 0.0
            features[f"{feat}_std"] = 0.0
    
    # Time features
    last_track_time = pd.to_datetime(window_df.iloc[-1]["played_at"])
    hour_sin, hour_cos = cyclical_encode(last_track_time.hour, 24)
    day_sin, day_cos = cyclical_encode(last_track_time.dayofweek, 7)
    
    features["hour_sin"] = hour_sin
    features["hour_cos"] = hour_cos
    features["day_sin"] = day_sin
    features["day_cos"] = day_cos
    features["is_weekend"] = 1 if last_track_time.weekday() >= 5 else 0
    
    # Session context
    features["session_position"] = len(session_df)
    features["session_length"] = len(session_df)
    features["time_since_session_start"] = 0.0  # Simplified
    
    # Current track features
    for feat in audio_features:
        val = window_df.iloc[-1][feat]
        features[f"current_{feat}"] = float(val) if pd.notna(val) else 0.0
    
    # Create feature vector in correct order
    X = np.array([[features.get(col, 0.0) for col in feature_cols]])
    
    # Predict
    pred_cluster = int(model.predict(X)[0])
    pred_proba = float(model.predict_proba(X)[0][pred_cluster])
    
    # Load cluster metadata
    with open(mood_clusters_path, 'r') as f:
        clusters_data = json.load(f)
    
    cluster_info = next(
        (c for c in clusters_data["clusters"] if c["cluster_id"] == pred_cluster),
        None
    )
    
    mood_label = cluster_info["label"] if cluster_info else f"Cluster {pred_cluster}"
    
    return {
        "mood_cluster_id": pred_cluster,
        "mood_label": mood_label,
        "confidence": pred_proba,
        "cluster_centroid": cluster_info.get("centroid") if cluster_info else None
    }

