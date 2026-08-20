import pandas as pd
import json
import pickle
from typing import Dict

try:
    from features.mood_prediction_features import build_feature_vector, build_mood_features
    from utils.listening_time import to_listening_time
except ModuleNotFoundError:
    from ..features.mood_prediction_features import build_feature_vector, build_mood_features
    from ..utils.listening_time import to_listening_time

def load_model(model_path: str) -> tuple:
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data["model"], model_data["feature_cols"], model_data.get("metadata", {})


def build_next_track_prediction(history_path: str, mood_model_path: str, 
                                mood_clusters_path: str, window_size: int = 3) -> Dict:
    # Load history
    df = pd.read_parquet(history_path)
    df = df.dropna(subset=["mood_cluster_id", "session_id"])
    if df.empty:
        raise ValueError("No mood-labeled listening history is available")
    df["played_at"] = to_listening_time(df["played_at"])
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
    model, feature_cols, model_metadata = load_model(mood_model_path)
    features = build_mood_features(
        window_df,
        session_position=len(session_df),
        session_start_time=session_df.iloc[0]["played_at"],
    )
    X = build_feature_vector(features, feature_cols)

    prediction_source = "model"
    class_probabilities = {}
    if model_metadata.get("model_beats_baseline") is False:
        prediction_source = f"{model_metadata.get('fallback_strategy', 'majority')}_baseline"
        if model_metadata.get("fallback_strategy") == "persistence":
            pred_cluster = int(window_df.iloc[-1]["mood_cluster_id"])
        else:
            pred_cluster = int(model_metadata.get("fallback_cluster", 0))
        pred_proba = float(model_metadata.get("fallback_validation_accuracy", 0.0))
        class_probabilities = {
            int(cluster): float(probability)
            for cluster, probability in model_metadata.get("class_priors", {}).items()
        }
    else:
        pred_cluster = int(model.predict(X)[0])
        probabilities = model.predict_proba(X)[0]
        class_probabilities = {
            int(cluster): float(probability)
            for cluster, probability in zip(model.classes_, probabilities)
        }
        pred_proba = class_probabilities[pred_cluster]
    
    # Load cluster metadata
    with open(mood_clusters_path, 'r') as f:
        clusters_data = json.load(f)
    
    cluster_info = next(
        (c for c in clusters_data["clusters"] if c["cluster_id"] == pred_cluster),
        None
    )
    
    mood_label = cluster_info["label"] if cluster_info else f"Cluster {pred_cluster}"
    mood_distribution = [
        {
            "mood_cluster_id": int(cluster_id),
            "mood_label": next(
                (
                    cluster["label"]
                    for cluster in clusters_data["clusters"]
                    if cluster["cluster_id"] == cluster_id
                ),
                f"Cluster {cluster_id}",
            ),
            "probability": float(probability),
        }
        for cluster_id, probability in sorted(
            class_probabilities.items(), key=lambda item: item[1], reverse=True
        )
    ]
    
    return {
        "mood_cluster_id": pred_cluster,
        "mood_label": mood_label,
        "confidence": pred_proba,
        "cluster_centroid": cluster_info.get("centroid") if cluster_info else None,
        "model_version": model_metadata.get("version_hash"),
        "prediction_source": prediction_source,
        "confidence_kind": (
            "validation_accuracy"
            if prediction_source.endswith("baseline")
            else "uncalibrated_model_probability"
        ),
        "mood_distribution": mood_distribution,
    }
