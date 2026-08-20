import pandas as pd
import numpy as np
import json
import pickle
from typing import Dict

try:
    from features.mood_prediction_features import build_feature_vector, build_mood_features
    from models.contextual_transition import persistence_probabilities, predict_contextual_transition
    from models.prediction_evaluation import (
        apply_temperature,
        combine_switch_and_mood_probabilities,
        describe_predictability,
    )
    from utils.listening_time import to_listening_time
except ModuleNotFoundError:
    from ..features.mood_prediction_features import build_feature_vector, build_mood_features
    from ..models.contextual_transition import persistence_probabilities, predict_contextual_transition
    from ..models.prediction_evaluation import (
        apply_temperature,
        combine_switch_and_mood_probabilities,
        describe_predictability,
    )
    from ..utils.listening_time import to_listening_time

def load_model(model_path: str) -> tuple:
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    metadata = dict(model_data.get("metadata", {}))
    if model_data.get("switch_model") is not None:
        metadata["_switch_model"] = model_data["switch_model"]
    return model_data["model"], model_data["feature_cols"], metadata


def select_probability_vector(model, model_metadata: Dict, features: Dict, X) -> tuple:
    """Return the quality-gated probability vector for model and online exports."""
    classes = np.asarray(model.classes_, dtype=int)
    priors = np.asarray(
        [model_metadata.get("class_priors", {}).get(str(int(value)), 0.0) for value in classes],
        dtype=float,
    )
    if priors.sum() <= 0:
        priors = np.full(len(classes), 1.0 / len(classes))
    else:
        priors /= priors.sum()

    selected_strategy = model_metadata.get("selected_strategy")
    if not selected_strategy:
        selected_strategy = (
            model_metadata.get("fallback_strategy", "majority")
            if model_metadata.get("model_beats_baseline") is False
            else "model"
        )
    context_level = None
    if selected_strategy == "contextual_transition" and model_metadata.get("transition_baseline"):
        values, levels = predict_contextual_transition(
            pd.DataFrame([features]), model_metadata["transition_baseline"]
        )
        probabilities = values[0]
        context_level = levels[0]
    elif selected_strategy == "persistence":
        probabilities = persistence_probabilities(
            pd.DataFrame([features]),
            classes,
            priors,
            float(model_metadata.get("persistence_strength", 0.85)),
        )[0]
    elif selected_strategy == "majority":
        probabilities = priors
    else:
        mood_probabilities = apply_temperature(
            model.predict_proba(X), float(model_metadata.get("temperature", 1.0))
        )
        if selected_strategy == "two_stage_model" and model_metadata.get("_switch_model"):
            switch_probability = apply_temperature(
                model_metadata["_switch_model"].predict_proba(X),
                float(model_metadata.get("switch_temperature", 1.0)),
            )[:, 1]
            probabilities = combine_switch_and_mood_probabilities(
                pd.DataFrame([features]),
                switch_probability,
                mood_probabilities,
                classes,
            )[0]
        else:
            probabilities = mood_probabilities[0]
            selected_strategy = "model"
    return classes, probabilities, selected_strategy, context_level


def build_next_track_prediction(history_path: str, mood_model_path: str, 
                                mood_clusters_path: str, window_size: int = 3) -> Dict:
    # Load history
    df = pd.read_parquet(history_path)
    df = df.dropna(subset=["mood_cluster_id", "session_id"])
    if df.empty:
        raise ValueError("No mood-labeled listening history is available")
    df["played_at"] = to_listening_time(df["played_at"])
    df = df.sort_values(["session_id", "played_at"]).reset_index(drop=True)

    # Load the predictor contract before selecting context. New models can use a
    # longer sequence while older deployed models remain compatible.
    model, feature_cols, model_metadata = load_model(mood_model_path)
    effective_window_size = int(model_metadata.get("window_size", window_size))

    # Get most recent session
    latest_session_id = df["session_id"].max()
    session_df = df[df["session_id"] == latest_session_id].copy()
    
    if len(session_df) < effective_window_size:
        # Use last N tracks across sessions
        session_df = df.tail(effective_window_size).copy()
    
    # Get last window_size tracks
    window_df = session_df.tail(effective_window_size).copy()
    features = build_mood_features(
        window_df,
        session_position=len(session_df),
        session_start_time=session_df.iloc[0]["played_at"],
    )
    X = build_feature_vector(features, feature_cols)

    classes, probabilities, selected_strategy, context_level = select_probability_vector(
        model, model_metadata, features, X
    )

    pred_cluster = int(classes[int(np.argmax(probabilities))])
    pred_proba = float(np.max(probabilities))
    class_probabilities = {
        int(cluster): float(probability)
        for cluster, probability in zip(classes, probabilities)
    }
    prediction_source = (
        selected_strategy
        if selected_strategy in {"model", "two_stage_model"}
        else f"{selected_strategy}_baseline"
    )
    predictability = describe_predictability(
        probabilities, model_metadata.get("abstention_policy", {})
    )
    
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
    last_observed_cluster = int(window_df.iloc[-1]["mood_cluster_id"])
    switch_probability = float(
        sum(
            probability
            for cluster_id, probability in class_probabilities.items()
            if cluster_id != last_observed_cluster
        )
    )
    
    return {
        "mood_cluster_id": pred_cluster,
        "mood_label": mood_label,
        "confidence": pred_proba,
        "cluster_centroid": cluster_info.get("centroid") if cluster_info else None,
        "model_version": model_metadata.get("version_hash"),
        "prediction_source": prediction_source,
        "confidence_kind": (
            "estimated_baseline_probability"
            if selected_strategy not in {"model", "two_stage_model"}
            else "calibrated_model_probability"
        ),
        "validation_reliability": model_metadata.get("selected_validation_accuracy"),
        "validation_log_loss": model_metadata.get("selected_validation_log_loss"),
        "predictability": predictability["level"],
        "normalized_entropy": predictability["normalized_entropy"],
        "abstained": predictability["abstained"],
        "context_level": context_level,
        "switch_probability": switch_probability,
        "recent_window_size": effective_window_size,
        "mood_distribution": mood_distribution,
    }
