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


def run_update():
    print("=" * 60)
    print("Starting Update Pipeline")
    print("=" * 60)
    
    try:
        # Step 1: Ingest new tracks, get metadata, get features
        print("\n[1/6] Ingesting new Spotify data (metadata and features)")
        df = ingest()
        
        if df.empty:
            print("No new tracks found. Exiting successfully.")
            return
        
        num_tracks = len(df)
        num_sessions = df["session_id"].nunique() if "session_id" in df.columns else 0
        
        # Get the latest ingestion file path
        latest_ingestion_file = get_latest_ingestion_file()
        if not latest_ingestion_file:
            print("Warning: Could not find ingestion file. Exiting.")
            return
        
        # Step 2: Assign clusters to tracks in ingestion file
        print("\n[2/6] Assigning new tracks to existing mood clusters")
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
        
        # Step 4: Update history.parquet with no null values
        print("\n[4/5] Updating history.parquet (removing null values)")
        update_history_from_ingestion(latest_ingestion_file)
        
        # Step 5: Build and export dashboard data
        print("\n[5/5] Building and exporting dashboard data")
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

