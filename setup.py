import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ingestion.spotify_ingest import ingest, update_history_from_ingestion
from features.build_mood_clusters import build_mood_clusters
from features.build_mood_clusters_incremental import assign_tracks_to_existing_clusters
from features.build_mood_dataset import build_mood_dataset
from features.build_session_dataset import build_session_dataset
from models.train_mood_model import train_mood_model
from models.train_session_model import train_session_model
from features.build_mood_clusters_incremental import get_latest_ingestion_file


def setup():
    print("=" * 60)
    print("Starting Initial Setup")
    print("=" * 60)
    
    try:
        # Step 1: Ingest data from Spotify
        print("\n[1/6] Ingesting initial data from Spotify")
        df = ingest()
        
        if df.empty:
            print("Error: No data was ingested. Please check your Spotify API credentials.")
            return
        
        print(f"Ingested {len(df)} tracks")
        
        # Get the ingestion file path and update history
        latest_ingestion_file = get_latest_ingestion_file()
        if not latest_ingestion_file:
            print("Error: Could not find ingestion file. Setup cannot continue.")
            return
        
        print(f"Updating history.parquet with ingested data")
        update_history_from_ingestion(latest_ingestion_file)
        
        # Step 2: Build mood clusters
        print("\n[2/7] Building mood clusters")
        build_mood_clusters()
        
        # Step 3: Assign cluster IDs to all tracks in history
        print("\n[3/7] Assigning cluster IDs to all tracks")
        try:
            assign_tracks_to_existing_clusters()
        except Exception as e:
            print(f"Warning: Could not assign clusters to all tracks: {e}")
            print("Continuing with cluster assignments from build_mood_clusters")
        
        # Step 4: Build mood prediction dataset
        print("\n[3/6] Building mood prediction dataset")
        build_mood_dataset()
        
        # Step 5: Build session start dataset
        print("\n[5/7] Building session start dataset")
        build_session_dataset()
        
        # Step 6: Train mood classifier
        print("\n[6/7] Training mood classifier")
        mood_metrics = train_mood_model()
        print(f"Mood model - Train Accuracy: {mood_metrics['train_accuracy']:.4f}, "
              f"Val Accuracy: {mood_metrics['val_accuracy']:.4f}")
        
        # Step 7: Train session classifier
        print("\n[7/7] Training session classifier")
        session_metrics = train_session_model()
        print(f"Session model - Train Accuracy: {session_metrics['train_accuracy']:.4f}, "
              f"Val Accuracy: {session_metrics['val_accuracy']:.4f}")
        
        print("\n" + "=" * 60)
        print("Initial Setup Completed Successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during setup: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    setup()




