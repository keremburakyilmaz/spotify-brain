import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, Optional
from collections import Counter
from scipy.stats import entropy


def calculate_hour_entropy(session_starts: pd.DataFrame) -> float:
    if len(session_starts) == 0:
        return 0.0
    
    hour_counts = session_starts["hour"].value_counts()
    probabilities = hour_counts.values / hour_counts.values.sum()
    
    # Calculate Shannon entropy
    hour_entropy = entropy(probabilities, base=2)
    
    return float(hour_entropy)


def detect_feature_drift(
    history_path: str = "data/history.parquet",
    days_recent: int = 7,
    days_prior: int = 30
) -> Dict:
    if not os.path.exists(history_path):
        return {
            "error": "History file not found",
            "drift_detected": False
        }
    
    df = pd.read_parquet(history_path)
    
    if df.empty or "session_id" not in df.columns:
        return {
            "error": "History is empty or missing session_id",
            "drift_detected": False
        }
    
    df["played_at"] = pd.to_datetime(df["played_at"])
    
    # Get date ranges
    max_date = df["played_at"].max().date()
    recent_start = max_date - timedelta(days=days_recent)
    prior_start = recent_start - timedelta(days=days_prior)
    prior_end = recent_start
    
    # Recent period (last N days)
    recent_df = df[df["played_at"].dt.date >= recent_start]
    
    # Prior period (M days before recent)
    prior_df = df[
        (df["played_at"].dt.date >= prior_start) &
        (df["played_at"].dt.date < prior_end)
    ]
    
    drift_results = {}
    
    # 1. Session duration drift
    def get_session_durations(df_period):
        session_starts = df_period.groupby("session_id")["played_at"].min()
        session_ends = df_period.groupby("session_id")["played_at"].max()
        durations = (session_ends - session_starts).dt.total_seconds() / 60  # minutes
        return durations.values
    
    recent_durations = get_session_durations(recent_df)
    prior_durations = get_session_durations(prior_df)
    
    if len(recent_durations) > 0 and len(prior_durations) > 0:
        recent_dur_mean = float(np.mean(recent_durations))
        recent_dur_std = float(np.std(recent_durations))
        prior_dur_mean = float(np.mean(prior_durations))
        prior_dur_std = float(np.std(prior_durations))
        
        dur_mean_pct_change = ((recent_dur_mean - prior_dur_mean) / max(prior_dur_mean, 1.0)) * 100
        dur_std_pct_change = ((recent_dur_std - prior_dur_std) / max(prior_dur_std, 1.0)) * 100
        
        drift_results["session_duration"] = {
            "recent_mean": recent_dur_mean,
            "recent_std": recent_dur_std,
            "prior_mean": prior_dur_mean,
            "prior_std": prior_dur_std,
            "mean_pct_change": float(dur_mean_pct_change),
            "std_pct_change": float(dur_std_pct_change)
        }
    else:
        drift_results["session_duration"] = {
            "recent_mean": 0.0,
            "recent_std": 0.0,
            "prior_mean": 0.0,
            "prior_std": 0.0,
            "mean_pct_change": 0.0,
            "std_pct_change": 0.0
        }
    
    # 2. Sessions per day
    def get_sessions_per_day(df_period):
        session_starts = df_period.groupby("session_id")["played_at"].min()
        daily_counts = session_starts.dt.date.value_counts()
        return daily_counts.values
    
    recent_sessions_per_day = get_sessions_per_day(recent_df)
    prior_sessions_per_day = get_sessions_per_day(prior_df)
    
    if len(recent_sessions_per_day) > 0 and len(prior_sessions_per_day) > 0:
        recent_sess_mean = float(np.mean(recent_sessions_per_day))
        recent_sess_std = float(np.std(recent_sessions_per_day))
        prior_sess_mean = float(np.mean(prior_sessions_per_day))
        prior_sess_std = float(np.std(prior_sessions_per_day))
        
        sess_mean_pct_change = ((recent_sess_mean - prior_sess_mean) / max(prior_sess_mean, 1.0)) * 100
        sess_std_pct_change = ((recent_sess_std - prior_sess_std) / max(prior_sess_std, 1.0)) * 100
        
        drift_results["sessions_per_day"] = {
            "recent_mean": recent_sess_mean,
            "recent_std": recent_sess_std,
            "prior_mean": prior_sess_mean,
            "prior_std": prior_sess_std,
            "mean_pct_change": float(sess_mean_pct_change),
            "std_pct_change": float(sess_std_pct_change)
        }
    else:
        drift_results["sessions_per_day"] = {
            "recent_mean": 0.0,
            "recent_std": 0.0,
            "prior_mean": 0.0,
            "prior_std": 0.0,
            "mean_pct_change": 0.0,
            "std_pct_change": 0.0
        }
    
    # 3. Tracks per day
    recent_tracks_per_day = recent_df.groupby(recent_df["played_at"].dt.date).size().values
    prior_tracks_per_day = prior_df.groupby(prior_df["played_at"].dt.date).size().values
    
    if len(recent_tracks_per_day) > 0 and len(prior_tracks_per_day) > 0:
        recent_tracks_mean = float(np.mean(recent_tracks_per_day))
        recent_tracks_std = float(np.std(recent_tracks_per_day))
        prior_tracks_mean = float(np.mean(prior_tracks_per_day))
        prior_tracks_std = float(np.std(prior_tracks_per_day))
        
        tracks_mean_pct_change = ((recent_tracks_mean - prior_tracks_mean) / max(prior_tracks_mean, 1.0)) * 100
        tracks_std_pct_change = ((recent_tracks_std - prior_tracks_std) / max(prior_tracks_std, 1.0)) * 100
        
        drift_results["tracks_per_day"] = {
            "recent_mean": recent_tracks_mean,
            "recent_std": recent_tracks_std,
            "prior_mean": prior_tracks_mean,
            "prior_std": prior_tracks_std,
            "mean_pct_change": float(tracks_mean_pct_change),
            "std_pct_change": float(tracks_std_pct_change)
        }
    else:
        drift_results["tracks_per_day"] = {
            "recent_mean": 0.0,
            "recent_std": 0.0,
            "prior_mean": 0.0,
            "prior_std": 0.0,
            "mean_pct_change": 0.0,
            "std_pct_change": 0.0
        }
    
    # 4. Start hour entropy
    recent_session_starts = recent_df.groupby("session_id")["played_at"].min().reset_index()
    recent_session_starts["hour"] = recent_session_starts["played_at"].dt.hour
    
    prior_session_starts = prior_df.groupby("session_id")["played_at"].min().reset_index()
    prior_session_starts["hour"] = prior_session_starts["played_at"].dt.hour
    
    recent_entropy = calculate_hour_entropy(recent_session_starts)
    prior_entropy = calculate_hour_entropy(prior_session_starts)
    
    entropy_change = recent_entropy - prior_entropy
    entropy_pct_change = ((recent_entropy - prior_entropy) / max(prior_entropy, 0.1)) * 100 if prior_entropy > 0 else 0.0
    
    drift_results["start_hour_entropy"] = {
        "recent": recent_entropy,
        "prior": prior_entropy,
        "change": float(entropy_change),
        "pct_change": float(entropy_pct_change)
    }
    
    # Flag significant drift (>20% change in key metrics)
    drift_threshold = 20.0  # 20% change threshold
    
    significant_drift = (
        abs(drift_results["session_duration"]["mean_pct_change"]) > drift_threshold or
        abs(drift_results["sessions_per_day"]["mean_pct_change"]) > drift_threshold or
        abs(drift_results["tracks_per_day"]["mean_pct_change"]) > drift_threshold or
        abs(drift_results["start_hour_entropy"]["pct_change"]) > drift_threshold
    )
    
    drift_results["drift_summary"] = {
        "drift_detected": significant_drift,
        "threshold": drift_threshold,
        "max_pct_change": float(max(
            abs(drift_results["session_duration"]["mean_pct_change"]),
            abs(drift_results["sessions_per_day"]["mean_pct_change"]),
            abs(drift_results["tracks_per_day"]["mean_pct_change"]),
            abs(drift_results["start_hour_entropy"]["pct_change"])
        ))
    }
    
    return drift_results


def detect_label_drift(
    history_path: str = "data/history.parquet",
    days_recent: int = 7,
    days_prior: int = 30
) -> Dict:
    if not os.path.exists(history_path):
        return {
            "error": "History file not found",
            "drift_detected": False
        }
    
    df = pd.read_parquet(history_path)
    
    if df.empty or "session_id" not in df.columns:
        return {
            "error": "History is empty or missing session_id",
            "drift_detected": False
        }
    
    df["played_at"] = pd.to_datetime(df["played_at"])
    
    # Get date ranges
    max_date = df["played_at"].max().date()
    recent_start = max_date - timedelta(days=days_recent)
    prior_start = recent_start - timedelta(days=days_prior)
    prior_end = recent_start
    
    # Recent period
    recent_df = df[df["played_at"].dt.date >= recent_start]
    
    # Prior period
    prior_df = df[
        (df["played_at"].dt.date >= prior_start) &
        (df["played_at"].dt.date < prior_end)
    ]
    
    drift_results = {}
    
    # 1. Session-start rate per day
    recent_session_starts = recent_df.groupby("session_id")["played_at"].min()
    prior_session_starts = prior_df.groupby("session_id")["played_at"].min()
    
    recent_daily_counts = recent_session_starts.dt.date.value_counts()
    prior_daily_counts = prior_session_starts.dt.date.value_counts()
    
    recent_rate = float(recent_daily_counts.mean()) if len(recent_daily_counts) > 0 else 0.0
    prior_rate = float(prior_daily_counts.mean()) if len(prior_daily_counts) > 0 else 0.0
    
    rate_pct_change = ((recent_rate - prior_rate) / max(prior_rate, 1.0)) * 100 if prior_rate > 0 else 0.0
    
    drift_results["session_start_rate"] = {
        "recent": recent_rate,
        "prior": prior_rate,
        "pct_change": float(rate_pct_change)
    }
    
    # 2. Fraction of days with at least one session
    recent_days_with_sessions = len(recent_daily_counts) / max(days_recent, 1.0)
    prior_days_with_sessions = len(prior_daily_counts) / max(days_prior, 1.0)
    
    days_pct_change = ((recent_days_with_sessions - prior_days_with_sessions) / max(prior_days_with_sessions, 0.01)) * 100
    
    drift_results["days_with_sessions"] = {
        "recent": float(recent_days_with_sessions),
        "prior": float(prior_days_with_sessions),
        "pct_change": float(days_pct_change)
    }
    
    # 3. Per-hour start frequency changes
    recent_hour_counts = recent_session_starts.dt.hour.value_counts().sort_index()
    prior_hour_counts = prior_session_starts.dt.hour.value_counts().sort_index()
    
    per_hour_changes = {}
    for hour in range(24):
        recent_count = float(recent_hour_counts.get(hour, 0))
        prior_count = float(prior_hour_counts.get(hour, 0))
        
        pct_change = ((recent_count - prior_count) / max(prior_count, 1.0)) * 100 if prior_count > 0 else 0.0
        
        per_hour_changes[str(hour)] = {
            "recent": recent_count,
            "prior": prior_count,
            "pct_change": float(pct_change)
        }
    
    drift_results["per_hour_frequency"] = per_hour_changes
    
    drift_threshold = 20.0
    significant_drift = (
        abs(rate_pct_change) > drift_threshold or
        abs(days_pct_change) > drift_threshold
    )
    
    drift_results["drift_summary"] = {
        "drift_detected": significant_drift,
        "threshold": drift_threshold,
        "max_pct_change": float(max(abs(rate_pct_change), abs(days_pct_change)))
    }
    
    return drift_results


def detect_drift(
    history_path: str = "data/history.parquet",
    days_recent: int = 7,
    days_prior: int = 30
) -> Dict:
    feature_drift = detect_feature_drift(history_path, days_recent, days_prior)
    label_drift = detect_label_drift(history_path, days_recent, days_prior)
    
    overall_drift = (
        feature_drift.get("drift_summary", {}).get("drift_detected", False) or
        label_drift.get("drift_summary", {}).get("drift_detected", False)
    )
    
    return {
        "feature_drift": feature_drift,
        "label_drift": label_drift,
        "overall_drift_detected": overall_drift,
        "detection_date": datetime.utcnow().isoformat()
    }

