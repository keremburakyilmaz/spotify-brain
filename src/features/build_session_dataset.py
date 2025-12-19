import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict


def cyclical_encode(value: float, max_value: float) -> tuple:
    sin_val = np.sin(2 * np.pi * value / max_value)
    cos_val = np.cos(2 * np.pi * value / max_value)
    return sin_val, cos_val


def build_session_dataset(history_path: str = "data/history.parquet",
                         output_path: str = "data/processed/session_start_train.parquet") -> pd.DataFrame:
    # Load history
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"History file not found: {history_path}")
    
    df = pd.read_parquet(history_path)
    
    if df.empty:
        raise ValueError("History is empty")
    
    print(f"Building session dataset from {len(df)} tracks...")
    
    # Ensure played_at is datetime
    df["played_at"] = pd.to_datetime(df["played_at"])
    
    # Get date range
    min_date = df["played_at"].min().date()
    max_date = df["played_at"].max().date()
    
    # Find session starts (first track of each session)
    session_starts = df.groupby("session_id")["played_at"].min().reset_index()
    session_starts["date"] = session_starts["played_at"].dt.date
    session_starts["hour"] = session_starts["played_at"].dt.hour
    
    # Create set of (date, hour) combinations that have session starts
    session_start_hours = set(zip(session_starts["date"], session_starts["hour"]))
    
    # Calculate session durations
    session_durations = df.groupby("session_id").agg({
        "played_at": ["min", "max"]
    }).reset_index()
    session_durations.columns = ["session_id", "start_time", "end_time"]
    session_durations["duration_minutes"] = (
        session_durations["end_time"] - session_durations["start_time"]
    ).dt.total_seconds() / 60
    
    # Map session_id to start time and duration
    session_info = session_starts.merge(
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
    
    # Build dataset: one row per (date, hour) combination
    samples = []
    current_date = min_date
    
    while current_date <= max_date:
        for hour in range(24):
            # Time features
            dt = datetime.combine(current_date, datetime.min.time().replace(hour=hour))
            hour_sin, hour_cos = cyclical_encode(hour, 24)
            day_sin, day_cos = cyclical_encode(dt.weekday(), 7)
            
            features = {
                "date": current_date,
                "hour": hour,
                "hour_sin": hour_sin,
                "hour_cos": hour_cos,
                "day_sin": day_sin,
                "day_cos": day_cos,
                "day_of_month": dt.day,
                "is_weekend": 1 if dt.weekday() >= 5 else 0
            }
            
            # Historical patterns: rolling listening frequency
            # Sessions in this hour over last 7 days
            date_7d_ago = current_date - timedelta(days=7)
            date_30d_ago = current_date - timedelta(days=30)
            
            # Count sessions in this hour over last 7 days
            sessions_7d = [
                (d, h) for d, h in session_start_hours
                if date_7d_ago <= d < current_date and h == hour
            ]
            features["rolling_listening_frequency_7d"] = len(sessions_7d)
            
            # Count sessions in this hour over last 30 days
            sessions_30d = [
                (d, h) for d, h in session_start_hours
                if date_30d_ago <= d < current_date and h == hour
            ]
            features["rolling_listening_frequency_30d"] = len(sessions_30d)
            
            # Time since last session (in hours)
            # Find most recent session before this (date, hour)
            previous_sessions = [
                (d, h) for d, h in session_start_hours
                if d < current_date or (d == current_date and h < hour)
            ]
            
            if previous_sessions:
                last_date, last_hour = max(previous_sessions, key=lambda x: (x[0], x[1]))
                last_dt = datetime.combine(last_date, datetime.min.time().replace(hour=last_hour))
                time_diff = (dt - last_dt).total_seconds() / 3600
                features["time_since_last_session"] = float(time_diff)
            else:
                # If no previous session, use time since first session or cap at 7 days (168 hours)
                if session_start_hours:
                    first_date, first_hour = min(session_start_hours, key=lambda x: (x[0], x[1]))
                    first_dt = datetime.combine(first_date, datetime.min.time().replace(hour=first_hour))
                    time_diff = (dt - first_dt).total_seconds() / 3600
                    features["time_since_last_session"] = min(float(time_diff), 168.0)  # Cap at 7 days
                else:
                    features["time_since_last_session"] = 168.0  # Default to 7 days if no sessions at all
            
            # Aggregated stats: average session duration last 7 days
            sessions_in_7d = [
                sid for sid, info in session_info_dict.items()
                if date_7d_ago <= info["date"] < current_date
            ]
            
            if sessions_in_7d:
                durations = [session_info_dict[sid]["duration"] for sid in sessions_in_7d]
                features["avg_session_duration_last_7d"] = float(np.mean(durations))
            else:
                features["avg_session_duration_last_7d"] = 0.0
            
            # Total tracks last 7 days
            tracks_7d = df[
                (df["played_at"].dt.date >= date_7d_ago) &
                (df["played_at"].dt.date < current_date)
            ]
            features["total_tracks_last_7d"] = len(tracks_7d)
            
            # Total tracks last 30 days
            tracks_30d = df[
                (df["played_at"].dt.date >= date_30d_ago) &
                (df["played_at"].dt.date < current_date)
            ]
            features["total_tracks_last_30d"] = len(tracks_30d)
            
            # Target: 1 if any track started in this hour, else 0
            features["target_session_start"] = 1 if (current_date, hour) in session_start_hours else 0
            
            samples.append(features)
        
        current_date += timedelta(days=1)
    
    # Create DataFrame
    dataset_df = pd.DataFrame(samples)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset_df.to_parquet(output_path, index=False)
    
    print(f"Created session dataset with {len(dataset_df)} samples")
    print(f"Saved to {output_path}")
    
    return dataset_df

if __name__ == "__main__":
    build_session_dataset()






