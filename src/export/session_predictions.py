import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, timedelta
from typing import Dict


def cyclical_encode(value: float, max_value: float) -> tuple:
    sin_val = np.sin(2 * np.pi * value / max_value)
    cos_val = np.cos(2 * np.pi * value / max_value)
    return sin_val, cos_val


def load_model(model_path: str) -> tuple:
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data["model"], model_data["feature_cols"]


def build_session_probabilities(session_model_path: str, history_path: str = "data/history.parquet") -> Dict:
    # Load model
    model, feature_cols = load_model(session_model_path)
    
    # Load history to calculate historical patterns
    history_df = pd.read_parquet(history_path) if os.path.exists(history_path) else pd.DataFrame()
    
    # Get all session starts from history (for calculating historical patterns)
    session_starts_history = []
    session_info_dict = {}
    
    if not history_df.empty and "session_id" in history_df.columns:
        history_df["played_at"] = pd.to_datetime(history_df["played_at"])
        
        # Get session starts
        session_starts_df = history_df.groupby("session_id")["played_at"].min().reset_index()
        session_starts_df["date"] = session_starts_df["played_at"].dt.date
        session_starts_df["hour"] = session_starts_df["played_at"].dt.hour
        session_starts_history = set(zip(session_starts_df["date"], session_starts_df["hour"]))
        
        # Calculate session durations
        session_durations = history_df.groupby("session_id").agg({
            "played_at": ["min", "max"]
        }).reset_index()
        session_durations.columns = ["session_id", "start_time", "end_time"]
        session_durations["duration_minutes"] = (
            session_durations["end_time"] - session_durations["start_time"]
        ).dt.total_seconds() / 60
        
        # Merge to get session info with dates
        session_info = session_starts_df.merge(
            session_durations[["session_id", "duration_minutes"]],
            on="session_id",
            how="left"
        )
        session_info_dict = {
            row["session_id"]: {
                "date": row["date"],
                "hour": row["hour"],
                "duration": row["duration_minutes"]
            }
            for _, row in session_info.iterrows()
        }
    
    # Get today's date
    today = datetime.now().date()
    
    # Build features for each hour today
    probabilities = {}
    
    for hour in range(24):
        dt = datetime.combine(today, datetime.min.time().replace(hour=hour))
        
        # Build features
        features = {}
        hour_sin, hour_cos = cyclical_encode(hour, 24)
        day_sin, day_cos = cyclical_encode(dt.weekday(), 7)
        
        features["hour_sin"] = hour_sin
        features["hour_cos"] = hour_cos
        features["day_sin"] = day_sin
        features["day_cos"] = day_cos
        features["day_of_month"] = dt.day
        features["is_weekend"] = 1 if dt.weekday() >= 5 else 0
        
        # Historical patterns: rolling listening frequency
        date_7d_ago = today - timedelta(days=7)
        date_30d_ago = today - timedelta(days=30)
        
        # Count sessions in this hour over last 7 days
        sessions_7d = [
            (d, h) for d, h in session_starts_history
            if date_7d_ago <= d < today and h == hour
        ]
        features["rolling_listening_frequency_7d"] = len(sessions_7d)
        
        # Count sessions in this hour over last 30 days
        sessions_30d = [
            (d, h) for d, h in session_starts_history
            if date_30d_ago <= d < today and h == hour
        ]
        features["rolling_listening_frequency_30d"] = len(sessions_30d)
        
        # Time since last session (in hours)
        previous_sessions = [
            (d, h) for d, h in session_starts_history
            if d < today or (d == today and h < hour)
        ]
        
        if previous_sessions:
            last_date, last_hour = max(previous_sessions, key=lambda x: (x[0], x[1]))
            last_dt = datetime.combine(last_date, datetime.min.time().replace(hour=last_hour))
            time_diff = (dt - last_dt).total_seconds() / 3600
            features["time_since_last_session"] = float(time_diff)
        else:
            # If no previous session, cap at 7 days (168 hours)
            if session_starts_history:
                first_date, first_hour = min(session_starts_history, key=lambda x: (x[0], x[1]))
                first_dt = datetime.combine(first_date, datetime.min.time().replace(hour=first_hour))
                time_diff = (dt - first_dt).total_seconds() / 3600
                features["time_since_last_session"] = min(float(time_diff), 168.0)
            else:
                features["time_since_last_session"] = 168.0  # Default to 7 days
        
        # Average session duration last 7 days
        sessions_in_7d = [
            sid for sid, info in session_info_dict.items()
            if date_7d_ago <= info["date"] < today
        ]
        
        if sessions_in_7d:
            durations = [session_info_dict[sid]["duration"] for sid in sessions_in_7d]
            features["avg_session_duration_last_7d"] = float(np.mean(durations))
        else:
            features["avg_session_duration_last_7d"] = 0.0
        
        # Total tracks last 7 days
        if not history_df.empty:
            tracks_7d = history_df[
                (history_df["played_at"].dt.date >= date_7d_ago) &
                (history_df["played_at"].dt.date < today)
            ]
            features["total_tracks_last_7d"] = len(tracks_7d)
        else:
            features["total_tracks_last_7d"] = 0
        
        # Total tracks last 30 days
        if not history_df.empty:
            tracks_30d = history_df[
                (history_df["played_at"].dt.date >= date_30d_ago) &
                (history_df["played_at"].dt.date < today)
            ]
            features["total_tracks_last_30d"] = len(tracks_30d)
        else:
            features["total_tracks_last_30d"] = 0
        
        # Set default values for any remaining features
        for col in feature_cols:
            if col not in features:
                features[col] = 0.0
        
        # Create feature vector
        X = np.array([[features.get(col, 0.0) for col in feature_cols]])
        
        # Predict probability
        proba = float(model.predict_proba(X)[0][1])
        probabilities[str(hour)] = proba
    
    return probabilities


def build_actual_session_distribution(history_path: str = "data/history.parquet") -> Dict:
    if not os.path.exists(history_path):
        return {str(hour): 0.0 for hour in range(24)}
    
    df = pd.read_parquet(history_path)
    if df.empty or "session_id" not in df.columns:
        return {str(hour): 0.0 for hour in range(24)}
    
    # Ensure played_at is datetime
    df["played_at"] = pd.to_datetime(df["played_at"])
    
    # Find session starts (first track of each session)
    session_starts = df.groupby("session_id")["played_at"].min().reset_index()
    session_starts["hour"] = session_starts["played_at"].dt.hour
    
    # Count sessions per hour
    hour_counts = session_starts["hour"].value_counts().sort_index()
    
    # Initialize all hours to 0
    distribution = {str(hour): 0 for hour in range(24)}
    
    # Fill in actual counts
    for hour, count in hour_counts.items():
        distribution[str(hour)] = int(count)
    
    # Normalize by max count (so max becomes 1.0)
    max_count = max(distribution.values()) if distribution.values() else 1
    if max_count > 0:
        normalized = {hour: float(count) / max_count for hour, count in distribution.items()}
    else:
        normalized = {hour: 0.0 for hour in distribution.keys()}
    
    return normalized

