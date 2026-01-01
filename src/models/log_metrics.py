import json
import os
from datetime import datetime
from typing import Dict, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.sanitize_json import sanitize_dict, sanitize_json_file


def log_metrics(history_path: str = "metrics/metrics_history.json",
                run_at: Optional[datetime] = None,
                num_tracks: int = 0,
                num_sessions: int = 0,
                mood_model_metrics: Optional[Dict] = None,
                session_model_metrics: Optional[Dict] = None,
                drift_data: Optional[Dict] = None) -> None:
    if run_at is None:
        run_at = datetime.utcnow()
    
    # Load existing history
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
    else:
        history = []
    
    # If no model metrics provided, preserve the most recent ones from history
    preserved_mood_metrics = None
    preserved_session_metrics = None
    if mood_model_metrics is None or session_model_metrics is None:
        # Find the most recent entry with model metrics
        for hist_entry in reversed(history):
            if mood_model_metrics is None and "mood_model" in hist_entry and preserved_mood_metrics is None:
                preserved_mood_metrics = hist_entry.get("mood_model")
            if session_model_metrics is None and "session_model" in hist_entry and preserved_session_metrics is None:
                preserved_session_metrics = hist_entry.get("session_model")
            # Stop if we found both metrics we need
            if (mood_model_metrics is not None or preserved_mood_metrics is not None) and \
               (session_model_metrics is not None or preserved_session_metrics is not None):
                break
    
    # Build metrics entry
    entry = {
        "run_at": run_at.isoformat(),
        "num_tracks": num_tracks,
        "num_sessions": num_sessions
    }
    
    # Use provided metrics if available, otherwise use preserved ones
    if mood_model_metrics:
        # Extract all mood model metrics, preserving None for missing values
        entry["mood_model"] = {
            "train_accuracy": mood_model_metrics.get("train_accuracy"),
            "val_accuracy": mood_model_metrics.get("val_accuracy"),
            "train_f1_macro": mood_model_metrics.get("train_f1_macro"),
            "val_f1_macro": mood_model_metrics.get("val_f1_macro"),
            "train_roc_auc": mood_model_metrics.get("train_roc_auc"),
            "val_roc_auc": mood_model_metrics.get("val_roc_auc")
        }
    elif preserved_mood_metrics:
        # Use preserved metrics from history (already formatted)
        entry["mood_model"] = preserved_mood_metrics
    
    if session_model_metrics:
        # Extract all session model metrics, including evaluation results
        entry["session_model"] = {
            "train_roc_auc": session_model_metrics.get("train_roc_auc"),
            "val_roc_auc": session_model_metrics.get("val_roc_auc"),
            "train_accuracy": session_model_metrics.get("train_accuracy"),
            "val_accuracy": session_model_metrics.get("val_accuracy"),
            "n_features": session_model_metrics.get("n_features"),
            "n_train": session_model_metrics.get("n_train"),
            "n_val": session_model_metrics.get("n_val")
        }
        
        # Add PR-AUC and baseline metrics from evaluation
        evaluation = session_model_metrics.get("evaluation")
        if evaluation:
            model_metrics = evaluation.get("model_metrics", {})
            entry["session_model"]["val_pr_auc"] = model_metrics.get("pr_auc")
            entry["session_model"]["val_f1"] = model_metrics.get("f1")
            entry["session_model"]["calibration_error"] = model_metrics.get("calibration_error")
            
            # Baseline comparisons
            baseline_neg = evaluation.get("baseline_always_negative", {})
            baseline_hist = evaluation.get("baseline_historical_hour", {})
            entry["session_model"]["baseline_neg_pr_auc"] = baseline_neg.get("pr_auc")
            entry["session_model"]["baseline_hist_pr_auc"] = baseline_hist.get("pr_auc")
            
            # Comparison flags
            comparison = evaluation.get("comparison", {})
            entry["session_model"]["model_beats_neg"] = comparison.get("model_beats_neg")
            entry["session_model"]["model_beats_hist"] = comparison.get("model_beats_hist")
            
            # Per-hour metrics (summary)
            per_hour = evaluation.get("per_hour_metrics", {})
            entry["session_model"]["per_hour_metrics"] = per_hour
        
        # Add model metadata if available
        model_metadata = session_model_metrics.get("model_metadata")
        if model_metadata:
            entry["session_model"]["model_metadata"] = model_metadata
    elif preserved_session_metrics:
        # Use preserved metrics from history (already formatted)
        entry["session_model"] = preserved_session_metrics
    
    # Add drift data if provided
    if drift_data:
        entry["drift"] = drift_data
    
    # Append to history
    history.append(entry)
    
    # Sanitize NaN values before saving
    history = sanitize_dict(history)
    
    # Save
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Double-check: sanitize the file after writing
    sanitize_json_file(history_path)
    
    print(f"Logged metrics to {history_path}")


