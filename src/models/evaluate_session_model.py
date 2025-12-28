import pandas as pd
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve
from typing import Dict, Tuple, List
from collections import Counter


def predict_always_negative(y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    predictions = np.zeros_like(y_true)
    probabilities = np.zeros_like(y_true, dtype=float)
    return predictions, probabilities


def predict_historical_hour_prior(
    df: pd.DataFrame,
    y_true: np.ndarray,
    date_col: str = "date",
    hour_col: str = "hour",
    target_col: str = "target_session_start"
) -> Tuple[np.ndarray, np.ndarray]:
    # Calculate historical frequencies per hour
    # Use only training data to avoid leakage
    if date_col not in df.columns:
        # If no date column, use all data (fallback)
        hour_counts = df.groupby(hour_col)[target_col].agg(['sum', 'count']).reset_index()
        hour_counts.columns = [hour_col, 'sessions', 'total_days']
    else:
        # Use only dates before the latest date in the dataset
        max_date = df[date_col].max()
        historical_df = df[df[date_col] < max_date]
        
        if len(historical_df) == 0:
            # Fallback to all data if no historical data
            historical_df = df
        
        hour_counts = historical_df.groupby(hour_col)[target_col].agg(['sum', 'count']).reset_index()
        hour_counts.columns = [hour_col, 'sessions', 'total_days']
    
    # Calculate probability for each hour
    hour_probs = {}
    for _, row in hour_counts.iterrows():
        hour = int(row[hour_col])
        total_days = max(1, row['total_days'])  # Avoid division by zero
        prob = row['sessions'] / total_days
        hour_probs[hour] = float(prob)
    
    # Default probability if hour not seen
    default_prob = 0.0
    
    # Generate predictions based on hour
    probabilities = np.array([
        hour_probs.get(int(df.iloc[i][hour_col]), default_prob)
        for i in range(len(df))
    ])
    
    # Convert probabilities to binary predictions (threshold 0.5)
    predictions = (probabilities >= 0.5).astype(int)
    
    return predictions, probabilities


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    baseline_name: str = "model"
) -> Dict:
    metrics = {}
    
    # PR-AUC (primary metric for imbalanced data)
    try:
        pr_auc = average_precision_score(y_true, y_proba)
        metrics["pr_auc"] = float(pr_auc)
    except ValueError:
        metrics["pr_auc"] = 0.0
    
    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_proba)
        metrics["roc_auc"] = float(roc_auc)
    except ValueError:
        metrics["roc_auc"] = 0.0
    
    # F1 score
    try:
        f1 = f1_score(y_true, y_pred)
        metrics["f1"] = float(f1)
    except ValueError:
        metrics["f1"] = 0.0
    
    # F1 at different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    f1_at_threshold = {}
    for threshold in thresholds:
        try:
            y_pred_thresh = (y_proba >= threshold).astype(int)
            f1_thresh = f1_score(y_true, y_pred_thresh)
            f1_at_threshold[f"f1_at_{threshold:.1f}"] = float(f1_thresh)
        except ValueError:
            f1_at_threshold[f"f1_at_{threshold:.1f}"] = 0.0
    
    metrics["f1_at_threshold"] = f1_at_threshold
    
    # Calibration error (mean absolute error between predicted and actual probabilities)
    try:
        # Use 10 bins for calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba, n_bins=10, strategy='uniform'
        )
        calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        metrics["calibration_error"] = float(calibration_error)
    except (ValueError, ZeroDivisionError):
        metrics["calibration_error"] = 0.0
    
    # Counts
    metrics["n_samples"] = len(y_true)
    metrics["n_positive"] = int(y_true.sum())
    metrics["n_negative"] = int(len(y_true) - y_true.sum())
    metrics["positive_rate"] = float(y_true.mean())
    
    return metrics


def calculate_per_hour_metrics(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    hour_col: str = "hour"
) -> Dict:
    hour_buckets = {
        "morning": list(range(6, 12)),  # 6-11
        "afternoon": list(range(12, 18)),  # 12-17
        "evening": list(range(18, 24)),  # 18-23
        "night": list(range(0, 6))  # 0-5
    }
    
    per_hour_metrics = {}
    
    for bucket_name, hours in hour_buckets.items():
        # Get indices for this hour bucket
        mask = df[hour_col].isin(hours)
        
        if mask.sum() == 0:
            per_hour_metrics[bucket_name] = {
                "pr_auc": 0.0,
                "roc_auc": 0.0,
                "n_samples": 0,
                "n_positive": 0
            }
            continue
        
        y_true_bucket = y_true[mask]
        y_proba_bucket = y_proba[mask]
        
        # Check if we have both classes
        if len(np.unique(y_true_bucket)) < 2:
            per_hour_metrics[bucket_name] = {
                "pr_auc": 0.0,
                "roc_auc": 0.0,
                "n_samples": int(mask.sum()),
                "n_positive": int(y_true_bucket.sum())
            }
            continue
        
        # Calculate metrics
        try:
            pr_auc = average_precision_score(y_true_bucket, y_proba_bucket)
        except ValueError:
            pr_auc = 0.0
        
        try:
            roc_auc = roc_auc_score(y_true_bucket, y_proba_bucket)
        except ValueError:
            roc_auc = 0.0
        
        per_hour_metrics[bucket_name] = {
            "pr_auc": float(pr_auc),
            "roc_auc": float(roc_auc),
            "n_samples": int(mask.sum()),
            "n_positive": int(y_true_bucket.sum())
        }
    
    return per_hour_metrics


def evaluate_session_model(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    date_col: str = "date",
    hour_col: str = "hour",
    target_col: str = "target_session_start"
) -> Dict:
    results = {}
    
    # Evaluate main model
    model_metrics = calculate_metrics(y_true, y_pred, y_proba, "model")
    results["model_metrics"] = model_metrics
    
    # Evaluate always-negative baseline
    y_pred_neg, y_proba_neg = predict_always_negative(y_true)
    baseline_neg_metrics = calculate_metrics(y_true, y_pred_neg, y_proba_neg, "always_negative")
    results["baseline_always_negative"] = baseline_neg_metrics
    
    # Evaluate historical hour prior baseline
    y_pred_hist, y_proba_hist = predict_historical_hour_prior(
        df, y_true, date_col, hour_col, target_col
    )
    baseline_hist_metrics = calculate_metrics(y_true, y_pred_hist, y_proba_hist, "historical_hour")
    results["baseline_historical_hour"] = baseline_hist_metrics
    
    # Per-hour metrics
    per_hour_metrics = calculate_per_hour_metrics(df, y_true, y_proba, hour_col)
    results["per_hour_metrics"] = per_hour_metrics
    
    # Comparison summary
    comparison = {
        "model_pr_auc": model_metrics["pr_auc"],
        "baseline_neg_pr_auc": baseline_neg_metrics["pr_auc"],
        "baseline_hist_pr_auc": baseline_hist_metrics["pr_auc"],
        "model_beats_neg": model_metrics["pr_auc"] > baseline_neg_metrics["pr_auc"],
        "model_beats_hist": model_metrics["pr_auc"] > baseline_hist_metrics["pr_auc"],
        "model_roc_auc": model_metrics["roc_auc"],
        "baseline_neg_roc_auc": baseline_neg_metrics["roc_auc"],
        "baseline_hist_roc_auc": baseline_hist_metrics["roc_auc"]
    }
    results["comparison"] = comparison
    
    return results
