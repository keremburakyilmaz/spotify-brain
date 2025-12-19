import pandas as pd
import numpy as np
import json
import os
from typing import Tuple
from sklearn.metrics import pairwise_distances_argmin_min
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def assign_tracks_to_existing_clusters(history_path: str = "data/history.parquet",
                                       clusters_path: str = "models/mood_clusters.json") -> pd.DataFrame:
    # Load history
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"History file not found: {history_path}")
    
    if not os.path.exists(clusters_path):
        raise FileNotFoundError(f"Cluster metadata not found: {clusters_path}. Run full retrain first.")
    
    df = pd.read_parquet(history_path)
    
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
    df.to_parquet(history_path, index=False)
    
    print(f"Assigned {len(unassigned_clean)} tracks to existing clusters")
    
    return df


if __name__ == "__main__":
    assign_tracks_to_existing_clusters()

