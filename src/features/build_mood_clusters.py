import pandas as pd
import numpy as np
import json
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Dict, List, Tuple
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.sanitize_json import sanitize_dict, sanitize_json_file


def extract_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    features = ["valence", "energy", "danceability", "acousticness", 
                "instrumentalness", "tempo_norm"]
    
    # Filter out rows with missing features
    df_clean = df.dropna(subset=features)
    
    if len(df_clean) == 0:
        raise ValueError("No tracks with complete audio features")
    
    X = df_clean[features].values
    return X, df_clean.index


def find_optimal_k(X: np.ndarray, k_range: Tuple[int, int] = (3, 15)) -> int:
    k_min, k_max = k_range
    n_samples = len(X)
    
    k_max = min(k_max, n_samples // 10, 15)
    k_min = max(k_min, 2)
    
    if k_max < k_min:
        return max(k_min, 3)
    
    inertias = []
    silhouette_scores = []
    k_values = list(range(k_min, k_max + 1))
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        inertias.append(kmeans.inertia_)
        
        # Silhouette score requires at least 2 samples per cluster
        if n_samples >= k * 2:
            sil_score = silhouette_score(X, labels)
            silhouette_scores.append(sil_score)
        else:
            silhouette_scores.append(-1)
    
    # Use silhouette score if available, otherwise use elbow method
    if any(s > 0 for s in silhouette_scores):
        optimal_k_idx = np.argmax(silhouette_scores)
        optimal_k = k_values[optimal_k_idx]
    else:
        # Elbow method: find point of maximum curvature
        if len(inertias) >= 3:
            # Calculate rate of change
            deltas = np.diff(inertias)
            deltas2 = np.diff(deltas)
            # Find maximum second derivative (elbow)
            if len(deltas2) > 0:
                optimal_k_idx = np.argmax(deltas2) + 1
            else:
                optimal_k_idx = len(k_values) // 2
            optimal_k = k_values[min(optimal_k_idx, len(k_values) - 1)]
        else:
            optimal_k = k_values[len(k_values) // 2]
    
    return optimal_k


def generate_cluster_label(centroid: np.ndarray, feature_names: List[str]) -> str:
    features = ["valence", "energy", "danceability", "acousticness", 
                "instrumentalness", "tempo_norm"]
    
    valence, energy, danceability, acousticness, instrumentalness, tempo_norm = centroid
    
    # Build label components
    energy_level = "High energy" if energy > 0.6 else "Low energy" if energy < 0.4 else "Medium energy"
    valence_level = "positive" if valence > 0.6 else "negative" if valence < 0.4 else "neutral"
    acoustic_level = "acoustic" if acousticness > 0.5 else "electronic"
    
    # Combine into label
    label = f"{energy_level}, {valence_level}"
    
    if acousticness > 0.5:
        label += f", {acoustic_level}"
    
    return label


def build_mood_clusters(history_path: str = "data/history.parquet",
                       output_path: str = "models/mood_clusters.json",
                       k_range: Tuple[int, int] = (3, 15)) -> pd.DataFrame:
    # Load history
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"History file not found: {history_path}")
    
    df = pd.read_parquet(history_path)
    
    if df.empty:
        raise ValueError("History is empty")
    
    print(f"Building mood clusters from {len(df)} tracks")
    
    # Extract feature matrix
    X, valid_indices = extract_feature_matrix(df)
    
    if len(X) < 3:
        raise ValueError(f"Not enough tracks with complete features: {len(X)}")
    
    print(f"Using {len(X)} tracks with complete features")
    
    # Find optimal K
    print(f"Finding optimal K in range {k_range}")
    optimal_k = find_optimal_k(X, k_range)
    print(f"Optimal K: {optimal_k}")
    
    # Fit KMeans
    print("Fitting KMeans")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Assign cluster IDs to tracks
    df.loc[valid_indices, "mood_cluster_id"] = labels
    
    # Build cluster metadata
    features = ["valence", "energy", "danceability", "acousticness", 
                "instrumentalness", "tempo_norm"]
    
    clusters_metadata = []
    for cluster_id in range(optimal_k):
        centroid = kmeans.cluster_centers_[cluster_id]
        label = generate_cluster_label(centroid, features)
        
        cluster_data = {
            "cluster_id": int(cluster_id),
            "label": label,
            "centroid": {
                "valence": float(centroid[0]),
                "energy": float(centroid[1]),
                "danceability": float(centroid[2]),
                "acousticness": float(centroid[3]),
                "instrumentalness": float(centroid[4]),
                "tempo_norm": float(centroid[5])
            }
        }
        clusters_metadata.append(cluster_data)
    
    # Save metadata
    clusters_data = {
        "optimal_k": optimal_k,
        "clusters": clusters_metadata
    }
    
    # Sanitize NaN values before saving
    clusters_data = sanitize_dict(clusters_data)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(clusters_data, f, indent=2)
    
    # Double-check: sanitize the file after writing
    sanitize_json_file(output_path)
    
    print(f"Saved cluster metadata to {output_path}")
    
    # Update history
    df.to_parquet(history_path, index=False)
    print(f"Updated history with mood cluster assignments")
    
    return df

if __name__ == "__main__":
    build_mood_clusters()







