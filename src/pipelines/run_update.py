import sys
import os
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ingestion.spotify_ingest import ingest, update_history_from_ingestion
from features.build_mood_clusters_incremental import (
    assign_clusters_to_ingestion_file,
    get_latest_ingestion_file
)
from export.build_dashboard_json import build_dashboard_json


def cyclical_encode(value: float, max_value: float) -> tuple:
    sin_val = np.sin(2 * np.pi * value / max_value)
    cos_val = np.cos(2 * np.pi * value / max_value)
    return sin_val, cos_val


def predict_mood_for_tracks(ingestion_file_path: str,
                           history_path: str = "data/history.parquet",
                           mood_model_path: str = "models/mood_classifier.pkl",
                           window_size: int = 3) -> pd.DataFrame:
    if not os.path.exists(mood_model_path):
        print(f"Warning: Mood model not found at {mood_model_path}. Skipping mood predictions.")
        return pd.read_parquet(ingestion_file_path)
    
    if not os.path.exists(ingestion_file_path):
        raise FileNotFoundError(f"Ingestion file not found: {ingestion_file_path}")
    
    df = pd.read_parquet(ingestion_file_path)
    
    if df.empty:
        return df
    
    # Load model
    with open(mood_model_path, 'rb') as f:
        model_data = pickle.load(f)
        mood_model = model_data["model"]
        feature_cols = model_data["feature_cols"]
    
    # Load history to get context for predictions
    history_df = pd.read_parquet(history_path) if os.path.exists(history_path) else pd.DataFrame()
    
    # Sort by session and time
    df = df.sort_values(["session_id", "played_at"]).reset_index(drop=True)
    
    audio_features = ["valence", "energy", "danceability", "acousticness", 
                     "instrumentalness", "tempo_norm"]
    
    # Predict mood for each track
    predicted_moods = []
    
    for idx, track in df.iterrows():
        session_id = track["session_id"]
        
        # Get context: last window_size tracks before this track (not including this track)
        # First, get tracks from history in the same session
        session_tracks_history = history_df[history_df["session_id"] == session_id].copy() if not history_df.empty else pd.DataFrame()
        
        # Get tracks from ingestion file in the same session (BEFORE current track)
        session_tracks_ingestion = df[df["session_id"] == session_id].iloc[:idx].copy()
        
        # Combine and sort
        if not session_tracks_history.empty:
            all_session_tracks = pd.concat([session_tracks_history, session_tracks_ingestion], ignore_index=True)
            all_session_tracks = all_session_tracks.sort_values("played_at").reset_index(drop=True)
        else:
            all_session_tracks = session_tracks_ingestion.copy()
        
        # Get last window_size tracks (or fewer if not enough)
        window_df = all_session_tracks.tail(window_size).copy()
        
        if len(window_df) < window_size:
            # Not enough context - use cluster assignment from step 2
            predicted_moods.append(track.get("mood_cluster_id", None))
            continue
        
        # Check if all tracks in window have mood_cluster_id
        if window_df["mood_cluster_id"].isna().any():
            # Not enough context - use cluster assignment from step 2
            predicted_moods.append(track.get("mood_cluster_id", None))
            continue
        
        # Build features (same as in build_mood_dataset)
        features = {}
        
        # Sequence features: mood_cluster_id of last N tracks
        for j, (_, row) in enumerate(window_df.iterrows()):
            features[f"mood_cluster_{j}"] = int(row["mood_cluster_id"])
        
        # Aggregated audio features from last N tracks
        for feat in audio_features:
            values = window_df[feat].dropna().values
            if len(values) > 0:
                features[f"{feat}_mean"] = float(np.mean(values))
                features[f"{feat}_std"] = float(np.std(values))
            else:
                features[f"{feat}_mean"] = 0.0
                features[f"{feat}_std"] = 0.0
        
        # Time features (use the current track's time, not the window's last track)
        current_track_time = pd.to_datetime(track["played_at"])
        hour_sin, hour_cos = cyclical_encode(current_track_time.hour, 24)
        day_sin, day_cos = cyclical_encode(current_track_time.dayofweek, 7)
        
        features["hour_sin"] = hour_sin
        features["hour_cos"] = hour_cos
        features["day_sin"] = day_sin
        features["day_cos"] = day_cos
        features["is_weekend"] = 1 if current_track_time.weekday() >= 5 else 0
        
        # Session context (include current track in count)
        features["session_position"] = len(all_session_tracks) + 1
        features["session_length"] = len(all_session_tracks) + 1
        
        # Time since session start
        if len(all_session_tracks) > 0:
            session_start = pd.to_datetime(all_session_tracks.iloc[0]["played_at"])
            time_diff = (current_track_time - session_start).total_seconds() / 60
            features["time_since_session_start"] = float(time_diff)
        else:
            features["time_since_session_start"] = 0.0
        
        # Current track features (use the current track being predicted)
        for feat in audio_features:
            val = track[feat]
            features[f"current_{feat}"] = float(val) if pd.notna(val) else 0.0
        
        # Create feature vector in correct order
        X = np.array([[features.get(col, 0.0) for col in feature_cols]])
        
        # Predict
        pred_cluster = int(mood_model.predict(X)[0])
        predicted_moods.append(pred_cluster)
    
    # Update mood_cluster_id with predictions (only where we made predictions)
    for idx, pred_mood in enumerate(predicted_moods):
        if pred_mood is not None:
            df.loc[idx, "mood_cluster_id"] = pred_mood
    
    # Save updated ingestion file
    df.to_parquet(ingestion_file_path, index=False)
    print(f"Predicted mood for {sum(p is not None for p in predicted_moods)} tracks")
    
    return df


def predict_session_for_tracks(ingestion_file_path: str,
                              history_path: str = "data/history.parquet",
                              session_model_path: str = "models/session_classifier.pkl") -> pd.DataFrame:
    if not os.path.exists(session_model_path):
        print(f"Warning: Session model not found at {session_model_path}. Skipping session predictions.")
        return pd.read_parquet(ingestion_file_path)
    
    if not os.path.exists(ingestion_file_path):
        raise FileNotFoundError(f"Ingestion file not found: {ingestion_file_path}")
    
    df = pd.read_parquet(ingestion_file_path)
    
    if df.empty:
        return df
    
    # Load model
    with open(session_model_path, 'rb') as f:
        model_data = pickle.load(f)
        session_model = model_data["model"]
        feature_cols = model_data["feature_cols"]
    
    # Load history to calculate historical patterns
    history_df = pd.read_parquet(history_path) if os.path.exists(history_path) else pd.DataFrame()
    
    # Ensure played_at is datetime
    df["played_at"] = pd.to_datetime(df["played_at"])
    if not history_df.empty:
        history_df["played_at"] = pd.to_datetime(history_df["played_at"])
    
    # Get all session starts from history (for calculating historical patterns)
    session_starts_history = []
    if not history_df.empty and "session_id" in history_df.columns:
        session_starts_df = history_df.groupby("session_id")["played_at"].min().reset_index()
        session_starts_df["date"] = session_starts_df["played_at"].dt.date
        session_starts_df["hour"] = session_starts_df["played_at"].dt.hour
        session_starts_history = set(zip(session_starts_df["date"], session_starts_df["hour"]))
    
    # Calculate session durations from history
    session_info_dict = {}
    if not history_df.empty and "session_id" in history_df.columns:
        session_starts_df = history_df.groupby("session_id")["played_at"].min().reset_index()
        session_starts_df["date"] = session_starts_df["played_at"].dt.date
        session_starts_df["hour"] = session_starts_df["played_at"].dt.hour
        
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
    
    # Get unique (date, hour) combinations from new tracks
    df["date"] = df["played_at"].dt.date
    df["hour"] = df["played_at"].dt.hour
    unique_hours = df[["date", "hour"]].drop_duplicates()
    
    predicted_probs = []
    
    # Predict for each unique (date, hour) combination
    for _, row in unique_hours.iterrows():
        current_date = row["date"]
        current_hour = row["hour"]
        dt = datetime.combine(current_date, datetime.min.time().replace(hour=current_hour))
        
        # Build features (same as in build_session_dataset.py)
        features = {}
        
        # Time features
        hour_sin, hour_cos = cyclical_encode(current_hour, 24)
        day_sin, day_cos = cyclical_encode(dt.weekday(), 7)
        
        features["hour_sin"] = hour_sin
        features["hour_cos"] = hour_cos
        features["day_sin"] = day_sin
        features["day_cos"] = day_cos
        features["day_of_month"] = dt.day
        features["is_weekend"] = 1 if dt.weekday() >= 5 else 0
        
        # Historical patterns: rolling listening frequency
        date_7d_ago = current_date - timedelta(days=7)
        date_30d_ago = current_date - timedelta(days=30)
        
        # Count sessions in this hour over last 7 days
        sessions_7d = [
            (d, h) for d, h in session_starts_history
            if date_7d_ago <= d < current_date and h == current_hour
        ]
        features["rolling_listening_frequency_7d"] = len(sessions_7d)
        
        # Count sessions in this hour over last 30 days
        sessions_30d = [
            (d, h) for d, h in session_starts_history
            if date_30d_ago <= d < current_date and h == current_hour
        ]
        features["rolling_listening_frequency_30d"] = len(sessions_30d)
        
        # Time since last session (in hours)
        previous_sessions = [
            (d, h) for d, h in session_starts_history
            if d < current_date or (d == current_date and h < current_hour)
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
            if date_7d_ago <= info["date"] < current_date
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
                (history_df["played_at"].dt.date < current_date)
            ]
            features["total_tracks_last_7d"] = len(tracks_7d)
        else:
            features["total_tracks_last_7d"] = 0
        
        # Total tracks last 30 days
        if not history_df.empty:
            tracks_30d = history_df[
                (history_df["played_at"].dt.date >= date_30d_ago) &
                (history_df["played_at"].dt.date < current_date)
            ]
            features["total_tracks_last_30d"] = len(tracks_30d)
        else:
            features["total_tracks_last_30d"] = 0
        
        # Create feature vector in correct order
        X = np.array([[features.get(col, 0.0) for col in feature_cols]])
        
        # Predict probability of session start
        proba = float(session_model.predict_proba(X)[0][1])  # Probability of class 1 (session start)
        predicted_probs.append({
            "date": current_date,
            "hour": current_hour,
            "session_start_probability": proba
        })
    
    # Map predictions back to tracks
    prob_dict = {(row["date"], row["hour"]): row["session_start_probability"] 
                 for _, row in pd.DataFrame(predicted_probs).iterrows()}
    
    df["session_start_probability"] = df.apply(
        lambda row: prob_dict.get((row["date"], row["hour"]), 0.0), axis=1
    )
    
    # Clean up temporary columns
    df = df.drop(columns=["date", "hour"], errors='ignore')
    
    # Save updated ingestion file
    df.to_parquet(ingestion_file_path, index=False)
    print(f"Predicted session start probabilities for {len(unique_hours)} unique (date, hour) combinations")
    
    return df


def run_update():
    print("=" * 60)
    print("Starting Update Pipeline")
    print("=" * 60)
    
    try:
        # Step 1: Ingest new tracks, get metadata, get features
        print("\n[1/5] Ingesting new Spotify data (metadata and features)")
        df = ingest()
        
        if df.empty:
            print("Warning: No new data available. Pipeline cannot continue.")
            return
        
        num_tracks = len(df)
        num_sessions = df["session_id"].nunique() if "session_id" in df.columns else 0
        
        # Get the latest ingestion file path
        latest_ingestion_file = get_latest_ingestion_file()
        if not latest_ingestion_file:
            print("Warning: Could not find ingestion file. Pipeline cannot continue.")
            return
        
        # Step 2: Assign clusters to tracks in ingestion file
        print("\n[2/5] Assigning new tracks to existing mood clusters")
        try:
            assign_clusters_to_ingestion_file(latest_ingestion_file)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print("Cluster metadata not found. Run full retrain first to initialize clusters.")
            return
        
        # Step 3: Predict mood clusters for new tracks
        print("\n[3/5] Predicting mood clusters for new tracks")
        try:
            predict_mood_for_tracks(latest_ingestion_file)
        except Exception as e:
            print(f"Warning: Could not predict mood: {e}")
            print("Continuing without mood predictions")
        
        # Step 4: Predict session start probabilities
        print("\n[4/5] Predicting session start probabilities")
        try:
            predict_session_for_tracks(latest_ingestion_file, history_path="data/history.parquet")
        except Exception as e:
            print(f"Warning: Could not predict session: {e}")
            print("Continuing without session predictions")
        
        # Step 5: Update history.parquet with no null values
        print("\n[5/6] Updating history.parquet (removing null values)")
        update_history_from_ingestion(latest_ingestion_file)
        
        # Step 6: Build and export dashboard data
        print("\n[6/6] Building and exporting dashboard data")
        try:
            build_dashboard_json()
        except Exception as e:
            print(f"Warning: Could not build dashboard: {e}")
            print("Continuing without dashboard update")
        
        print("\n" + "=" * 60)
        print(f"Update Pipeline Completed Successfully")
        print(f"Processed {num_tracks} tracks from {num_sessions} sessions")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError in pipeline: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_update()

