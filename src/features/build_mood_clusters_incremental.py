import pandas as pd
import numpy as np
import json
import os
from typing import Tuple, Optional
from sklearn.metrics import pairwise_distances_argmin_min
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def assign_tracks_to_existing_clusters(history_path: str = "data/history.parquet",
                                       clusters_path: str = "models/mood_clusters.json",
                                       ingestions_dir: str = "data/ingestions") -> pd.DataFrame:
    # Load history
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"History file not found: {history_path}")
    
    if not os.path.exists(clusters_path):
        raise FileNotFoundError(f"Cluster metadata not found: {clusters_path}. Run full retrain first.")
    
    df = pd.read_parquet(history_path)
    
    # Also check ingestion files for tracks that might have been dropped from history.parquet
    if os.path.exists(ingestions_dir):
        ingestion_files = [f for f in os.listdir(ingestions_dir) 
                         if f.startswith("ingestion_") and f.endswith(".parquet")]
        if ingestion_files:
            ingestion_files.sort()
            latest_ingestion = os.path.join(ingestions_dir, ingestion_files[-1])
            ingestion_df = pd.read_parquet(latest_ingestion)
            if not ingestion_df.empty:
                # Merge with history, keeping ingestion tracks even if they have some nulls
                if df.empty:
                    df = ingestion_df.copy()
                else:
                    # Add missing columns
                    for col in df.columns:
                        if col not in ingestion_df.columns:
                            ingestion_df[col] = None
                    for col in ingestion_df.columns:
                        if col not in df.columns:
                            df[col] = None
                    # Combine and deduplicate
                    df = pd.concat([df, ingestion_df], ignore_index=True)
                    df = df.drop_duplicates(subset=["track_id", "played_at"], keep="last")
    
    if df.empty:
        raise ValueError("History is empty")
    
    # Load existing cluster centroids
    with open(clusters_path, 'r') as f:
        clusters_data = json.load(f)
    
    centroids = np.array([
        [
            c["centroid"]["valence"],
            c["centroid"]["energy"],
            c["centroid"]["danceability"],
            c["centroid"]["acousticness"],
            c["centroid"]["instrumentalness"],
            c["centroid"]["tempo_norm"]
        ]
        for c in sorted(clusters_data["clusters"], key=lambda x: x["cluster_id"])
    ])
    
    # Find tracks that need cluster assignment
    features = ["valence", "energy", "danceability", "acousticness", 
                "instrumentalness", "tempo_norm"]
    
    # Get tracks without cluster assignment or with missing features
    df["played_at"] = pd.to_datetime(df["played_at"])
    
    # Find unassigned tracks (no mood_cluster_id or missing features)
    # Check if mood_cluster_id column exists
    if "mood_cluster_id" not in df.columns:
        df["mood_cluster_id"] = None
    
    unassigned_mask = (
        df["mood_cluster_id"].isna() | 
        df[features].isna().any(axis=1)
    )
    
    unassigned_df = df[unassigned_mask].copy()
    
    if len(unassigned_df) == 0:
        print("No new tracks to assign clusters")
        return df
    
    # Filter to tracks with complete features
    unassigned_clean = unassigned_df.dropna(subset=features)
    
    if len(unassigned_clean) == 0:
        print("No tracks with complete features to assign")
        return df
    
    print(f"Assigning clusters to {len(unassigned_clean)} new tracks")
    
    # Extract feature matrix
    X_new = unassigned_clean[features].values
    
    # Assign to nearest cluster
    cluster_ids, distances = pairwise_distances_argmin_min(X_new, centroids, metric='euclidean')
    
    # Update mood_cluster_id
    df.loc[unassigned_clean.index, "mood_cluster_id"] = cluster_ids
    
    # Save updated history
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    df.to_parquet(history_path, index=False)
    
    print(f"Assigned {len(unassigned_clean)} tracks to existing clusters")
    
    return df


def assign_clusters_to_ingestion_file(ingestion_file_path: str,
                                      clusters_path: str = "models/mood_clusters.json") -> pd.DataFrame:
    """
    Assign clusters to tracks in the ingestion file and update the file.
    Returns the updated DataFrame.
    """
    if not os.path.exists(ingestion_file_path):
        raise FileNotFoundError(f"Ingestion file not found: {ingestion_file_path}")
    
    if not os.path.exists(clusters_path):
        raise FileNotFoundError(f"Cluster metadata not found: {clusters_path}. Run full retrain first.")
    
    df = pd.read_parquet(ingestion_file_path)
    
    if df.empty:
        print("Ingestion file is empty")
        return df
    
    # Load existing cluster centroids
    with open(clusters_path, 'r') as f:
        clusters_data = json.load(f)
    
    centroids = np.array([
        [
            c["centroid"]["valence"],
            c["centroid"]["energy"],
            c["centroid"]["danceability"],
            c["centroid"]["acousticness"],
            c["centroid"]["instrumentalness"],
            c["centroid"]["tempo_norm"]
        ]
        for c in sorted(clusters_data["clusters"], key=lambda x: x["cluster_id"])
    ])
    
    # Find tracks that need cluster assignment
    features = ["valence", "energy", "danceability", "acousticness", 
                "instrumentalness", "tempo_norm"]
    
    # Check if mood_cluster_id column exists
    if "mood_cluster_id" not in df.columns:
        df["mood_cluster_id"] = None
    
    # Find unassigned tracks with complete features
    unassigned_mask = (
        df["mood_cluster_id"].isna() | 
        df[features].isna().any(axis=1)
    )
    
    unassigned_df = df[unassigned_mask].copy()
    
    if len(unassigned_df) == 0:
        print("No new tracks to assign clusters in ingestion file")
        return df
    
    # Filter to tracks with complete features
    unassigned_clean = unassigned_df.dropna(subset=features)
    
    if len(unassigned_clean) == 0:
        print("No tracks with complete features to assign in ingestion file")
        return df
    
    print(f"Assigning clusters to {len(unassigned_clean)} tracks in ingestion file")
    
    # Extract feature matrix
    X_new = unassigned_clean[features].values
    
    # Assign to nearest cluster
    cluster_ids, distances = pairwise_distances_argmin_min(X_new, centroids, metric='euclidean')
    
    # Update mood_cluster_id
    df.loc[unassigned_clean.index, "mood_cluster_id"] = cluster_ids
    
    # Save updated ingestion file
    df.to_parquet(ingestion_file_path, index=False)
    
    print(f"Updated ingestion file with {len(unassigned_clean)} cluster assignments")
    
    return df


def get_latest_ingestion_file(ingestions_dir: str = "data/ingestions") -> Optional[str]:
    if not os.path.exists(ingestions_dir):
        return None
    
    ingestion_files = [f for f in os.listdir(ingestions_dir) 
                      if f.startswith("ingestion_") and f.endswith(".parquet")]
    
    if not ingestion_files:
        return None
    
    ingestion_files.sort()
    return os.path.join(ingestions_dir, ingestion_files[-1])


if __name__ == "__main__":
    assign_tracks_to_existing_clusters()

