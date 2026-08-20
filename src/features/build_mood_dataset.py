import numpy as np
import pandas as pd
import os

try:
    from features.mood_prediction_features import AUDIO_FEATURES, build_mood_features
    from utils.listening_time import to_listening_time
except ModuleNotFoundError:  # Package imports used by tests and library callers.
    from .mood_prediction_features import AUDIO_FEATURES, build_mood_features
    from ..utils.listening_time import to_listening_time


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
    
    print(f"Building mood dataset from {len(df)} tracks")
    
    # Spotify timestamps are UTC. Behavioral hour/day features must use the
    # listener's timezone consistently in training and inference.
    df["played_at"] = to_listening_time(df["played_at"])

    # Sort by session and time
    df = df.sort_values(["session_id", "played_at"]).reset_index(drop=True)
    
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
            if window_df[AUDIO_FEATURES].isna().any().any():
                continue

            features = build_mood_features(
                window_df,
                session_position=i + window_size,
                session_start_time=session_df.iloc[0]["played_at"],
            )

            # Metadata used only for leakage-safe temporal/grouped validation.
            features["session_id"] = int(session_id)
            features["target_played_at"] = next_track["played_at"]
            
            # Target: mood_cluster_id of next track
            features["target_mood_cluster"] = int(next_track["mood_cluster_id"])
            features["target_mood_switch"] = int(
                int(next_track["mood_cluster_id"]) != int(window_df.iloc[-1]["mood_cluster_id"])
            )
            features["target_energy_direction"] = int(
                np.sign(float(next_track["energy"]) - float(window_df.iloc[-1]["energy"]))
            )
            features["target_artist_repeat"] = int(
                "artist_name" in next_track.index
                and next_track["artist_name"] in set(window_df["artist_name"])
            )
            
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

