"""
Incremental pipeline.

Runs lightweight updates: ingest new data, assign clusters incrementally,
build features for new samples only, and retrain models on recent data.
For faster, frequent updates.
"""

import sys
import os
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ingestion.spotify_ingest import ingest
from features.build_mood_clusters_incremental import assign_tracks_to_existing_clusters
from features.build_mood_dataset_incremental import build_mood_dataset_incremental
from features.build_session_dataset_incremental import build_session_dataset_incremental
from models.train_mood_model_incremental import train_mood_model_incremental
from models.train_session_model_incremental import train_session_model_incremental
from models.log_metrics import log_metrics
from export.build_dashboard_json import build_dashboard_json


def run_incremental():
    """
    Run incremental pipeline with true incremental learning.
    
    - Assigns new tracks to existing clusters (no reclustering)
    - Only builds features for new tracks
    - Retrains models on recent data only
    """
    print("=" * 60)
    print("Starting Incremental Pipeline")
    print("=" * 60)
    
    try:
        # Step 1: Ingest new data only
        print("\n[1/8] Ingesting new Spotify data")
        df = ingest()
        
        if df.empty:
            print("Warning: No data available. Pipeline cannot continue.")
            return
        
        num_tracks = len(df)
        num_sessions = df["session_id"].nunique() if "session_id" in df.columns else 0
        
        # Step 2: Assign new tracks to existing clusters (incremental)
        print("\n[2/8] Assigning new tracks to existing mood clusters")
        try:
            assign_tracks_to_existing_clusters()
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print("Cluster metadata not found. Run full retrain first to initialize clusters.")
            return
        
        # Step 3: Build mood dataset incrementally (only new tracks)
        print("\n[3/8] Building mood prediction dataset (incremental)")
        build_mood_dataset_incremental()
        
        # Step 4: Build session dataset incrementally (only new dates)
        print("\n[4/8] Building session start dataset (incremental)")
        build_session_dataset_incremental()
        
        # Step 5: Continue training mood model with new data
        print("\n[5/8] Continuing training mood classifier with new data")
        mood_metrics = train_mood_model_incremental(new_trees=5, min_samples=1)
        
        # Step 6: Continue training session model with new data
        print("\n[6/8] Continuing training session classifier with new data")
        session_metrics = train_session_model_incremental(new_trees=5, min_samples=1)
        
        # Step 7: Log metrics
        print("\n[7/8] Logging metrics")
        log_metrics(
            num_tracks=num_tracks,
            num_sessions=num_sessions,
            mood_model_metrics=mood_metrics,
            session_model_metrics=session_metrics
        )
        
        # Step 8: Export dashboard data
        print("\n[8/8] Exporting dashboard data")
        build_dashboard_json()
        
        print("\n" + "=" * 60)
        print("Incremental Pipeline Completed Successfully")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError in pipeline: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_incremental()






