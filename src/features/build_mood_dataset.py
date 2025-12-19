import pandas as pd
import numpy as np
import os


def cyclical_encode(value: float, max_value: float) -> tuple:
    sin_val = np.sin(2 * np.pi * value / max_value)
    cos_val = np.cos(2 * np.pi * value / max_value)
    return sin_val, cos_val


def build_mood_dataset(history_path: str = "data/history.parquet",
                      output_path: str = "data/processed/mood_nexttrack_train.parquet",
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
    
    print(f"Building mood dataset from {len(df)} tracks...")
    
    # Sort by session and time
    df = df.sort_values(["session_id", "played_at"]).reset_index(drop=True)
    
    # Audio feature columns
    audio_features = ["valence", "energy", "danceability", "acousticness", 
                     "instrumentalness", "tempo_norm"]
    
    # Build dataset
    samples = []
    
    for session_id in df["session_id"].unique():
        session_df = df[df["session_id"] == session_id].copy()
        
        if len(session_df) < window_size + 1:
            continue  # Need at least window_size + 1 tracks for a sample
        
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
            
            samples.append(features)
    
    if not samples:
        raise ValueError("No valid samples created")
    
    # Create DataFrame
    dataset_df = pd.DataFrame(samples)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset_df.to_parquet(output_path, index=False)
    
    print(f"Created mood dataset with {len(dataset_df)} samples")
    print(f"Saved to {output_path}")
    
    return dataset_df

if __name__ == "__main__":
    build_mood_dataset()






