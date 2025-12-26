import sys
import os
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from features.build_mood_clusters import build_mood_clusters
from features.build_mood_dataset import build_mood_dataset
from features.build_session_dataset import build_session_dataset
from models.train_mood_model import train_mood_model
from models.train_session_model import train_session_model
from models.log_metrics import log_metrics
from export.build_dashboard_json import build_dashboard_json
import pandas as pd
import os


def run_full_retrain():
    print("=" * 60)
    print("Starting Full Retrain Pipeline")
    print("=" * 60)
    
    try:
        # Check if history exists
        history_path = "data/history.parquet"
        if not os.path.exists(history_path):
            print("Error: history.parquet not found. Run incremental pipeline first to ingest data.")
            return
        
        df = pd.read_parquet(history_path)
        if df.empty:
            print("Error: history.parquet is empty. Run incremental pipeline first to ingest data.")
            return
        
        num_tracks = len(df)
        num_sessions = df["session_id"].nunique() if "session_id" in df.columns else 0
        
        print(f"\n[1/7] Using existing history: {num_tracks} tracks, {num_sessions} sessions")
        
        print("\n[2/7] Building mood clusters")
        build_mood_clusters()
        
        print("\n[3/7] Building mood prediction dataset")
        build_mood_dataset()
        
        print("\n[4/7] Building session start dataset")
        build_session_dataset()
        
        print("\n[5/7] Training mood classifier")
        mood_metrics = train_mood_model()
        
        print("\n[6/7] Training session classifier")
        session_metrics = train_session_model()
        
        print("\n[7/7] Logging metrics and exporting dashboard data")
        log_metrics(
            num_tracks=num_tracks,
            num_sessions=num_sessions,
            mood_model_metrics=mood_metrics,
            session_model_metrics=session_metrics
        )
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






