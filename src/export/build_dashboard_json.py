import pandas as pd
import numpy as np
import json
import os
import pickle
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.sanitize_json import sanitize_dict, sanitize_json_file


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
    
    if len(df) < window_size:
        return {
            "mood_cluster_id": None,
            "mood_label": None,
            "confidence": 0.0
        }
    
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
        "confidence": pred_proba
    }


def build_recently_played(history_path: str, limit: int = 25) -> List[Dict]:
    if not os.path.exists(history_path):
        return []
    
    df = pd.read_parquet(history_path)
    if df.empty:
        return []
    
    # Ensure played_at is datetime
    df["played_at"] = pd.to_datetime(df["played_at"])
    
    # Sort by played_at descending and take most recent
    df = df.sort_values("played_at", ascending=False).head(limit)
    
    # Build list of track dictionaries
    recently_played = []
    for _, row in df.iterrows():
        track_data = {
            "track_id": str(row.get("track_id", "")),
            "track_name": str(row.get("track_name", "")) if pd.notna(row.get("track_name")) else "Unknown Track",
            "artist_name": str(row.get("artist_name", "")) if pd.notna(row.get("artist_name")) else "Unknown Artist",
            "played_at": row["played_at"].isoformat() if pd.notna(row["played_at"]) else None,
        }
        
        # Add image_url if available
        if "image_url" in row and pd.notna(row.get("image_url")):
            track_data["image_url"] = str(row["image_url"])
        else:
            track_data["image_url"] = None
        
        # Add mood_cluster_id if available
        if "mood_cluster_id" in row and pd.notna(row.get("mood_cluster_id")):
            track_data["mood_cluster_id"] = int(row["mood_cluster_id"])
        else:
            track_data["mood_cluster_id"] = None
        
        recently_played.append(track_data)
    
    return recently_played


def build_session_probabilities(session_model_path: str) -> Dict:
    # Load model
    model, feature_cols = load_model(session_model_path)
    
    # Get today's date
    today = datetime.now().date()
    
    # Build features for each hour today
    probabilities = {}
    
    for hour in range(24):
        dt = datetime.combine(today, datetime.min.time().replace(hour=hour))
        
        # Build features (simplified - would need full feature engineering in production)
        features = {}
        hour_sin, hour_cos = cyclical_encode(hour, 24)
        day_sin, day_cos = cyclical_encode(dt.weekday(), 7)
        
        features["hour_sin"] = hour_sin
        features["hour_cos"] = hour_cos
        features["day_sin"] = day_sin
        features["day_cos"] = day_cos
        features["day_of_month"] = dt.day
        features["is_weekend"] = 1 if dt.weekday() >= 5 else 0
        
        # Set default values for historical features (would need actual data)
        for col in feature_cols:
            if col not in features:
                features[col] = 0.0
        
        # Create feature vector
        X = np.array([[features.get(col, 0.0) for col in feature_cols]])
        
        # Predict probability
        proba = float(model.predict_proba(X)[0][1])
        probabilities[str(hour)] = proba
    
    return probabilities


def build_dashboard_json(history_path: str = "data/history.parquet",
                        mood_model_path: str = "models/mood_classifier.pkl",
                        session_model_path: str = "models/session_classifier.pkl",
                        mood_clusters_path: str = "models/mood_clusters.json",
                        metrics_path: str = "metrics/metrics_history.json",
                        output_path: str = "export/dashboard_data.json") -> Dict:
    print("Building dashboard data")
    
    # Load metadata
    with open(mood_clusters_path, 'r') as f:
        mood_clusters = json.load(f)
    
    # Load metrics
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics_history = json.load(f)
        latest_metrics = metrics_history[-1] if metrics_history else None
        # History should exclude the latest entry to avoid duplication
        if len(metrics_history) > 1:
            recent_metrics = metrics_history[-11:-1] if len(metrics_history) > 11 else metrics_history[:-1]
        else:
            recent_metrics = []
    else:
        latest_metrics = None
        recent_metrics = []
    
    # Build predictions
    print("Building next-track prediction")
    next_prediction = build_next_track_prediction(
        history_path, mood_model_path, mood_clusters_path
    )
    
    print("Building session probabilities")
    session_probs = build_session_probabilities(session_model_path)
    
    print("Building recently played tracks")
    recently_played = build_recently_played(history_path, limit=25)
    
    # Build mood trajectory (optional - last day aggregated into 15-min bins)
    mood_trajectory = None
    if os.path.exists(history_path):
        df = pd.read_parquet(history_path)
        if not df.empty:
            df["played_at"] = pd.to_datetime(df["played_at"])
            # Ensure yesterday is timezone-aware (UTC) to match played_at
            yesterday = datetime.now(timezone.utc) - timedelta(days=1)
            recent_df = df[df["played_at"] >= yesterday]
            
            if len(recent_df) > 0:
                # Aggregate into 15-minute bins
                recent_df["time_bin"] = recent_df["played_at"].dt.floor("15T")
                binned = recent_df.groupby("time_bin").agg({
                    "valence": "mean",
                    "energy": "mean"
                }).reset_index()
                
                mood_trajectory = [
                    {
                        "time": row["time_bin"].isoformat(),
                        "valence": float(row["valence"]) if pd.notna(row["valence"]) else 0.5,
                        "energy": float(row["energy"]) if pd.notna(row["energy"]) else 0.5
                    }
                    for _, row in binned.iterrows()
                ]
    
    # Build dashboard data structure
    dashboard_data = {
        "generated_at": datetime.utcnow().isoformat(),
        "next_prediction": next_prediction,
        "session_probs": session_probs,
        "mood_clusters": mood_clusters["clusters"],
        "mood_trajectory": mood_trajectory,
        "recently_played": recently_played,
        "metrics": {
            "latest": latest_metrics,
            "history": recent_metrics
        }
    }
    
    # Sanitize NaN values before saving
    dashboard_data = sanitize_dict(dashboard_data)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(dashboard_data, f, indent=2)
    
    # Double-check: sanitize the file after writing
    sanitize_json_file(output_path)
    
    print(f"Saved dashboard data to {output_path}")
    
    return dashboard_data


if __name__ == "__main__":
    build_dashboard_json()






