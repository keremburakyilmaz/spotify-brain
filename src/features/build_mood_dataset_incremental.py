import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import List


def cyclical_encode(value: float, max_value: float) -> tuple:
    sin_val = np.sin(2 * np.pi * value / max_value)
    cos_val = np.cos(2 * np.pi * value / max_value)
    return sin_val, cos_val


def build_mood_dataset_incremental(history_path: str = "data/history.parquet",
                                  dataset_path: str = "data/processed/mood_nexttrack_train.parquet",
                                  window_size: int = 3) -> pd.DataFrame:
    # Load history
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"History file not found: {history_path}")
    
    df = pd.read_parquet(history_path)
    
    if df.empty:
        raise ValueError("History is empty")
    
    # Filter to tracks with mood_cluster_id
    df = df.dropna(subset=["mood_cluster_id", "session_id"])
    
    if len(df) < window_size + 1:
        raise ValueError(f"Not enough tracks for window size {window_size}")
    
    # Load existing dataset to find last processed timestamp
    last_processed_timestamp = None
    existing_samples = []
    
    if os.path.exists(dataset_path):
        existing_df = pd.read_parquet(dataset_path)
        if not existing_df.empty and "last_track_timestamp" in existing_df.columns:
            last_processed_timestamp = existing_df["last_track_timestamp"].max()
            existing_samples = existing_df.to_dict('records')
            print(f"Found {len(existing_samples)} existing samples")
    
    # Sort by session and time
    df = df.sort_values(["session_id", "played_at"]).reset_index(drop=True)
    
    # If we have a last processed timestamp, only process tracks after it
    if last_processed_timestamp:
        df["played_at"] = pd.to_datetime(df["played_at"])
        df = df[df["played_at"] > pd.to_datetime(last_processed_timestamp)]
        print(f"Processing {len(df)} new tracks since last update")
    else:
        print(f"Building initial dataset from {len(df)} tracks")
    
    if len(df) == 0:
        print("No new tracks to process")
        if existing_samples:
            return pd.DataFrame(existing_samples)
        else:
            raise ValueError("No data available")
    
    # Audio feature columns
    audio_features = ["valence", "energy", "danceability", "acousticness", 
                     "instrumentalness", "tempo_norm"]
    
    # Build new samples
    new_samples = []
    
    for session_id in df["session_id"].unique():
        session_df = df[df["session_id"] == session_id].copy()
        
        if len(session_df) < window_size + 1:
            continue
        
        for i in range(len(session_df) - window_size):
            window_df = session_df.iloc[i:i+window_size]
            next_track = session_df.iloc[i + window_size]
            
            # Skip if any track in window is missing features
            if window_df[audio_features].isna().any().any():
                continue
            
            # Extract features
            features = {}
            
            # Sequence features: mood_cluster_id of last N tracks
            for j, track_idx in enumerate(window_df.index):
                features[f"mood_cluster_{j}"] = int(window_df.loc[track_idx, "mood_cluster_id"])
            
            # Aggregated audio features from last N tracks: mean and std
            for feat in audio_features:
                values = window_df[feat].values
                features[f"{feat}_mean"] = float(np.mean(values))
                features[f"{feat}_std"] = float(np.std(values))
            
            # Time features
            last_track_time = pd.to_datetime(window_df.iloc[-1]["played_at"])
            hour_sin, hour_cos = cyclical_encode(last_track_time.hour, 24)
            day_sin, day_cos = cyclical_encode(last_track_time.dayofweek, 7)
            
            features["hour_sin"] = hour_sin
            features["hour_cos"] = hour_cos
            features["day_sin"] = day_sin
            features["day_cos"] = day_cos
            features["is_weekend"] = 1 if last_track_time.weekday() >= 5 else 0
            
            # Session context
            features["session_position"] = i + window_size
            features["session_length"] = len(session_df)
            
            # Time since session start (in minutes)
            session_start = pd.to_datetime(session_df.iloc[0]["played_at"])
            time_diff = (last_track_time - session_start).total_seconds() / 60
            features["time_since_session_start"] = float(time_diff)
            
            # Current track features (most recent track in window)
            for feat in audio_features:
                features[f"current_{feat}"] = float(window_df.iloc[-1][feat])
            
            # Target: mood_cluster_id of next track
            features["target_mood_cluster"] = int(next_track["mood_cluster_id"])
            
            # Store timestamp for incremental processing
            features["last_track_timestamp"] = last_track_time.isoformat()
            
            new_samples.append(features)
    
    if not new_samples and not existing_samples:
        raise ValueError("No valid samples created")
    
    print(f"Created {len(new_samples)} new samples")
    
    # Combine existing and new samples
    all_samples = existing_samples + new_samples
    dataset_df = pd.DataFrame(all_samples)
    
    # Remove duplicates based on feature values (in case of re-runs)
    # Use a subset of key features to identify duplicates
    key_cols = [col for col in dataset_df.columns 
                if col not in ["last_track_timestamp", "target_mood_cluster"]]
    dataset_df = dataset_df.drop_duplicates(subset=key_cols, keep='last')
    
    # Save
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    dataset_df.to_parquet(dataset_path, index=False)
    
    print(f"Saved dataset with {len(dataset_df)} total samples to {dataset_path}")
    
    return dataset_df


if __name__ == "__main__":
    build_mood_dataset_incremental()

