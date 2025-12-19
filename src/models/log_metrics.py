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
                session_model_metrics: Optional[Dict] = None) -> None:
    if run_at is None:
        run_at = datetime.utcnow()
    
    # Load existing history
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
    else:
        history = []
    
    # Build metrics entry
    entry = {
        "run_at": run_at.isoformat(),
        "num_tracks": num_tracks,
        "num_sessions": num_sessions
    }
    
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
    
    if session_model_metrics:
        # Extract all session model metrics, preserving None for missing values
        entry["session_model"] = {
            "train_roc_auc": session_model_metrics.get("train_roc_auc"),
            "val_roc_auc": session_model_metrics.get("val_roc_auc"),
            "train_accuracy": session_model_metrics.get("train_accuracy"),
            "val_accuracy": session_model_metrics.get("val_accuracy")
        }
    
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


