import json
import os
from datetime import datetime
from typing import Dict
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.sanitize_json import sanitize_dict, sanitize_json_file
from export.mood_predictions import build_next_track_prediction
from export.recommendations import get_recommended_tracks
from export.session_predictions import build_session_probabilities, build_actual_session_distribution
from export.history_utils import build_recently_played, build_mood_trajectory
from ingestion.spotify_ingest import SpotifyIngester


def cyclical_encode(value: float, max_value: float) -> tuple:
    import numpy as np
    sin_val = np.sin(2 * np.pi * value / max_value)
    cos_val = np.cos(2 * np.pi * value / max_value)
    return sin_val, cos_val


def load_model(model_path: str) -> tuple:
    import pickle
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data["model"], model_data["feature_cols"]


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
    
    # Get recommended tracks based on predicted mood (3 tracks: centroid match + 2 variations)
    recommended_tracks = []
    if next_prediction.get("cluster_centroid") and next_prediction.get("mood_cluster_id") is not None:
        print("Fetching recommended tracks from ReccoBeats")
        try:
            spotify_ingester = SpotifyIngester()
            recommended_tracks = get_recommended_tracks(
                next_prediction.get("cluster_centroid"),
                next_prediction.get("mood_cluster_id"),
                history_path,
                spotify_ingester,
                num_tracks=3
            )
        except Exception as e:
            print(f"Warning: Could not fetch recommended tracks: {e}")
            recommended_tracks = []
    
    # Add recommended tracks to next_prediction
    if recommended_tracks:
        next_prediction["recommended_tracks"] = recommended_tracks
    
    print("Building session probabilities")
    session_probs = build_session_probabilities(session_model_path, history_path)
    
    # Build top hours (top 3 predicted hours with probabilities)
    top_hours = []
    if session_probs:
        # Sort hours by probability
        sorted_hours = sorted(session_probs.items(), key=lambda x: x[1], reverse=True)
        top_hours = [
            {
                "hour": int(hour),
                "probability": float(prob)
            }
            for hour, prob in sorted_hours[:3]
        ]
    
    print("Building actual session distribution")
    session_actual = build_actual_session_distribution(history_path)
    
    # Build drift data
    print("Detecting drift")
    drift_data = None
    try:
        from models.detect_drift import detect_drift
        drift_data = detect_drift(history_path)
    except Exception as e:
        print(f"Warning: Could not detect drift: {e}")
        drift_data = None
    
    print("Building recently played tracks")
    recently_played = build_recently_played(history_path, limit=25)
    
    print("Building mood trajectory")
    mood_trajectory = build_mood_trajectory(history_path)
    
    # Build dashboard data structure
    dashboard_data = {
        "generated_at": datetime.utcnow().isoformat(),
        "next_prediction": next_prediction,
        "session_probs": session_probs,
        "top_hours": top_hours,
        "session_actual": session_actual,
        "mood_clusters": mood_clusters["clusters"],
        "mood_trajectory": mood_trajectory,
        "recently_played": recently_played,
        "metrics": {
            "latest": latest_metrics,
            "history": recent_metrics
        }
    }
    
    # Add drift data if available
    if drift_data:
        dashboard_data["drift"] = drift_data
    
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
