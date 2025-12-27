import pandas as pd
import numpy as np
import os
import requests
import time
from typing import Dict, List, Optional
from ingestion.spotify_ingest import SpotifyIngester


def get_recommended_tracks(
    cluster_centroid: Optional[Dict],
    cluster_id: Optional[int],
    history_path: str = "data/history.parquet",
    spotify_ingester: Optional[SpotifyIngester] = None,
    num_tracks: int = 3
) -> List[Dict]:
    if not cluster_centroid:
        return []
    
    if not spotify_ingester:
        try:
            spotify_ingester = SpotifyIngester()
        except Exception as e:
            print(f"[ReccoBeats] Could not initialize Spotify ingester: {e}")
            return []
    
    # Get seed tracks: last 5 tracks from history that belong to this cluster
    seed_track_ids = []
    if cluster_id is not None and os.path.exists(history_path):
        try:
            history_df = pd.read_parquet(history_path)
            if not history_df.empty and "mood_cluster_id" in history_df.columns and "track_id" in history_df.columns:
                # Filter tracks from the same cluster
                cluster_tracks = history_df[history_df["mood_cluster_id"] == cluster_id].copy()
                if not cluster_tracks.empty:
                    # Sort by played_at descending and get last 5
                    if "played_at" in cluster_tracks.columns:
                        cluster_tracks["played_at"] = pd.to_datetime(cluster_tracks["played_at"])
                        cluster_tracks = cluster_tracks.sort_values("played_at", ascending=False)
                    
                    # Get unique track IDs (in case same track appears multiple times)
                    unique_tracks = cluster_tracks.drop_duplicates(subset=["track_id"])
                    seed_track_ids = unique_tracks["track_id"].head(5).tolist()
                    seed_track_ids = [str(tid) for tid in seed_track_ids if pd.notna(tid)]
        except Exception as e:
            print(f"[ReccoBeats] Warning: Could not get seed tracks from history: {e}")
    
    recommended_tracks = []
    base_url = "https://api.reccobeats.com/v1/track/recommendation"
    
    # Variation amounts for features (to get different tracks in the cluster)
    variations = [
        {},  # No variation for first track (centroid match)
        {"valence": 0.1, "energy": 0.08, "danceability": 0.1},  # Small variation
        {"valence": -0.1, "energy": -0.08, "danceability": -0.1}  # Opposite variation
    ]
    
    for i in range(num_tracks):
        try:
            # Build recommendation request using cluster centroid features
            params = {}
            
            # Apply variation to features for diversity
            variation = variations[i] if i < len(variations) else {}
            
            # Valence (0-1)
            if "valence" in cluster_centroid:
                valence = float(cluster_centroid["valence"])
                valence += variation.get("valence", 0.0)
                params["valence"] = max(0.0, min(1.0, valence))  # Clamp to [0, 1]
            
            # Energy (0-1)
            if "energy" in cluster_centroid:
                energy = float(cluster_centroid["energy"])
                energy += variation.get("energy", 0.0)
                params["energy"] = max(0.0, min(1.0, energy))
            
            # Danceability (0-1)
            if "danceability" in cluster_centroid:
                danceability = float(cluster_centroid["danceability"])
                danceability += variation.get("danceability", 0.0)
                params["danceability"] = max(0.0, min(1.0, danceability))
            
            # Acousticness (0-1)
            if "acousticness" in cluster_centroid:
                params["acousticness"] = float(cluster_centroid["acousticness"])
            
            # Instrumentalness (0-1)
            if "instrumentalness" in cluster_centroid:
                params["instrumentalness"] = float(cluster_centroid["instrumentalness"])
            
            # Tempo
            if "tempo_norm" in cluster_centroid:
                # Convert normalized tempo (0-1) back to BPM (assuming 0-250 BPM range)
                tempo_bpm = float(cluster_centroid["tempo_norm"]) * 250.0
                params["tempo"] = tempo_bpm
            elif "tempo" in cluster_centroid:
                params["tempo"] = float(cluster_centroid["tempo"])
            
            # Add size parameter
            params["size"] = 1
            
            # Add seeds parameter
            if seed_track_ids:
                params["seeds"] = ",".join(seed_track_ids)
            
            # Request recommendation
            resp = requests.get(base_url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            
            # Parse response
            recommended_track = None
            
            if isinstance(data, dict) and "content" in data:
                content = data["content"]
                if isinstance(content, list) and len(content) > 0:
                    recommended_track = content[0]
                else:
                    print(f"[ReccoBeats] Content is not a list or is empty for track {i+1}")
                    continue
            else:
                print(f"[ReccoBeats] Unexpected response format for track {i+1}")
                continue
            
            if not recommended_track:
                print(f"[ReccoBeats] No recommended track found in response for track {i+1}")
                continue
            
            # Extract Spotify track ID from ReccoBeats schema
            spotify_id = None
            
            # First try the "id" field (Spotify track ID)
            if "id" in recommended_track:
                spotify_id = str(recommended_track["id"])
            
            # Fallback: extract from "href" (Spotify URL)
            if not spotify_id:
                href = recommended_track.get("href", "")
                if href and "open.spotify.com/track/" in href:
                    try:
                        spotify_id = href.split("/track/")[1].split("?")[0].split("/")[0]
                    except Exception:
                        pass
            
            if not spotify_id:
                print(f"[ReccoBeats] Could not extract Spotify track ID from recommendation {i+1}")
                continue
            
            # Skip if we already have this track
            if any(t.get("track_id") == spotify_id for t in recommended_tracks):
                print(f"[ReccoBeats] Duplicate track {spotify_id}, skipping")
                continue
            
            # Fetch track metadata from Spotify
            metadata = spotify_ingester.fetch_track_metadata([spotify_id])
            track_metadata = metadata.get(spotify_id)
            
            if not track_metadata:
                print(f"[ReccoBeats] Could not fetch metadata for track {spotify_id}")
                continue
            
            recommended_tracks.append({
                "track_id": spotify_id,
                "track_name": track_metadata.get("track_name", ""),
                "artist_name": track_metadata.get("artist_name", ""),
                "image_url": track_metadata.get("image_url")
            })
            
            # Small delay between requests to avoid rate limiting
            if i < num_tracks - 1:
                time.sleep(0.2)
                
        except requests.RequestException as e:
            print(f"[ReccoBeats] Error fetching recommendation {i+1}: {e}")
            continue
        except Exception as e:
            print(f"[ReccoBeats] Unexpected error for recommendation {i+1}: {e}")
            continue
    
    return recommended_tracks

