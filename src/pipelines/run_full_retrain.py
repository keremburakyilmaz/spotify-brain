import sys
import os
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ingestion.spotify_ingest import ingest
from features.build_mood_clusters import build_mood_clusters
from features.build_mood_dataset import build_mood_dataset
from features.build_session_dataset import build_session_dataset
from models.train_mood_model import train_mood_model
from models.train_session_model import train_session_model
from models.log_metrics import log_metrics
from export.build_dashboard_json import build_dashboard_json


def run_full_retrain():
    print("=" * 60)
    print("Starting Full Retrain Pipeline")
    print("=" * 60)
    
    try:
        print("\n[1/8] Ingesting Spotify data")
        df = ingest()
        
        if df.empty:
            print("Warning: No data available. Pipeline cannot continue.")
            return
        
        num_tracks = len(df)
        num_sessions = df["session_id"].nunique() if "session_id" in df.columns else 0
        
        print("\n[2/8] Building mood clusters")
        build_mood_clusters()
        
        print("\n[3/8] Building mood prediction dataset")
        build_mood_dataset()
        
        print("\n[4/8] Building session start dataset")
        build_session_dataset()
        
        print("\n[5/8] Training mood classifier")
        mood_metrics = train_mood_model()
        
        print("\n[6/8] Training session classifier")
        session_metrics = train_session_model()
        
        print("\n[7/8] Logging metrics")
        log_metrics(
            num_tracks=num_tracks,
            num_sessions=num_sessions,
            mood_model_metrics=mood_metrics,
            session_model_metrics=session_metrics
        )
        
        print("\n[8/8] Exporting dashboard data")
        build_dashboard_json()
        
        print("\n" + "=" * 60)
        print("Full Retrain Pipeline Completed Successfully")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError in pipeline: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_full_retrain()






