import pandas as pd
import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict


def build_recently_played(history_path: str, limit: int = 25) -> List[Dict]:
    if not os.path.exists(history_path):
        return []
    
    df = pd.read_parquet(history_path)
    if df.empty:
        return []
    
    # Ensure played_at is datetime
    df["played_at"] = pd.to_datetime(df["played_at"])
    
    # Sort by played_at descending and take most recent
    df = df.sort_values("played_at", ascending=False).head(limit)
    
    # Build list of track dictionaries
    recently_played = []
    for _, row in df.iterrows():
        track_data = {
            "track_id": str(row.get("track_id", "")),
            "track_name": str(row.get("track_name", "")) if pd.notna(row.get("track_name")) else "Unknown Track",
            "artist_name": str(row.get("artist_name", "")) if pd.notna(row.get("artist_name")) else "Unknown Artist",
            "played_at": row["played_at"].isoformat() if pd.notna(row["played_at"]) else None,
        }
        
        # Add image_url if available
        if "image_url" in row and pd.notna(row.get("image_url")):
            track_data["image_url"] = str(row["image_url"])
        else:
            track_data["image_url"] = None
        
        # Add mood_cluster_id if available
        if "mood_cluster_id" in row and pd.notna(row.get("mood_cluster_id")):
            track_data["mood_cluster_id"] = int(row["mood_cluster_id"])
        else:
            track_data["mood_cluster_id"] = None
        
        recently_played.append(track_data)
    
    return recently_played


def build_mood_trajectory(history_path: str) -> List[Dict]:
    if not os.path.exists(history_path):
        return None
    
    df = pd.read_parquet(history_path)
    if df.empty:
        return None
    
    df["played_at"] = pd.to_datetime(df["played_at"])
    # Ensure yesterday is timezone-aware (UTC) to match played_at
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    recent_df = df[df["played_at"] >= yesterday]
    
    if len(recent_df) == 0:
        return None
    
    # Aggregate into 15-minute bins
    recent_df["time_bin"] = recent_df["played_at"].dt.floor("15T")
    binned = recent_df.groupby("time_bin").agg({
        "valence": "mean",
        "energy": "mean"
    }).reset_index()
    
    mood_trajectory = [
        {
            "time": row["time_bin"].isoformat(),
            "valence": float(row["valence"]) if pd.notna(row["valence"]) else 0.5,
            "energy": float(row["energy"]) if pd.notna(row["energy"]) else 0.5
        }
        for _, row in binned.iterrows()
    ]
    
    return mood_trajectory

