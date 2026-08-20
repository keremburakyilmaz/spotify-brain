import sys
import os
import pandas as pd
import pickle

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ingestion.spotify_ingest import ingest_with_metadata, update_history_from_ingestion
from features.build_mood_clusters_incremental import (
    assign_clusters_to_ingestion_file,
)
from features.mood_prediction_features import build_feature_vector, build_mood_features
from export.build_dashboard_json import build_dashboard_json
from models.log_metrics import log_metrics
from utils.listening_time import to_listening_time


def predict_mood_for_tracks(ingestion_file_path: str,
                           history_path: str = "data/history.parquet",
                           mood_model_path: str = "models/mood_classifier.pkl",
                           window_size: int = 3) -> pd.DataFrame:
    if not os.path.exists(mood_model_path):
        print(f"Warning: Mood model not found at {mood_model_path}. Skipping mood predictions.")
        return pd.read_parquet(ingestion_file_path)
    
    if not os.path.exists(ingestion_file_path):
        raise FileNotFoundError(f"Ingestion file not found: {ingestion_file_path}")
    
    df = pd.read_parquet(ingestion_file_path)
    
    if df.empty:
        return df
    
    # Load model. Predictions are stored separately for online evaluation and
    # must never replace the audio-derived mood_cluster_id ground truth.
    with open(mood_model_path, 'rb') as f:
        model_data = pickle.load(f)
        mood_model = model_data["model"]
        feature_cols = model_data["feature_cols"]
    
    # Load history to get context for predictions
    history_df = pd.read_parquet(history_path) if os.path.exists(history_path) else pd.DataFrame()
    
    df = df.sort_values(["session_id", "played_at"]).reset_index(drop=True)
    feature_df = df.copy()
    feature_df["played_at"] = to_listening_time(feature_df["played_at"])
    if not history_df.empty:
        history_df = history_df.copy()
        history_df["played_at"] = to_listening_time(history_df["played_at"])
    
    # Predict mood for each track
    predicted_moods = []
    predicted_confidences = []
    
    for idx, track in feature_df.iterrows():
        session_id = track["session_id"]
        
        # Get context: last window_size tracks before this track (not including this track)
        # First, get tracks from history in the same session
        if not history_df.empty:
            session_tracks_history = history_df[
                (history_df["session_id"] == session_id)
                & (history_df["played_at"] < track["played_at"])
            ].copy()
        else:
            session_tracks_history = pd.DataFrame()
        
        # Get tracks from ingestion file in the same session (BEFORE current track)
        session_tracks_ingestion = feature_df[
            feature_df["session_id"] == session_id
        ]
        session_tracks_ingestion = session_tracks_ingestion[
            session_tracks_ingestion["played_at"] < track["played_at"]
        ].copy()
        
        # Combine and sort
        if not session_tracks_history.empty:
            all_session_tracks = pd.concat([session_tracks_history, session_tracks_ingestion], ignore_index=True)
            all_session_tracks = all_session_tracks.drop_duplicates(
                subset=["track_id", "played_at"], keep="last"
            )
            all_session_tracks = all_session_tracks.sort_values("played_at").reset_index(drop=True)
        else:
            all_session_tracks = session_tracks_ingestion.copy()
        
        # Get last window_size tracks (or fewer if not enough)
        window_df = all_session_tracks.tail(window_size).copy()
        
        if len(window_df) < window_size:
            predicted_moods.append(None)
            predicted_confidences.append(None)
            continue
        
        # Check if all tracks in window have mood_cluster_id
        if window_df["mood_cluster_id"].isna().any():
            predicted_moods.append(None)
            predicted_confidences.append(None)
            continue

        features = build_mood_features(
            window_df,
            session_position=len(all_session_tracks),
            session_start_time=all_session_tracks.iloc[0]["played_at"],
        )
        X = build_feature_vector(features, feature_cols)
        
        # Predict
        pred_cluster = int(mood_model.predict(X)[0])
        predicted_moods.append(pred_cluster)
        class_index = list(mood_model.classes_).index(pred_cluster)
        predicted_confidences.append(
            float(mood_model.predict_proba(X)[0][class_index])
        )

    df["predicted_mood_cluster_id"] = pd.array(predicted_moods, dtype="Int64")
    df["prediction_confidence"] = pd.array(
        predicted_confidences, dtype="Float64"
    )
    
    # Save updated ingestion file
    df.to_parquet(ingestion_file_path, index=False)
    print(
        "Recorded next-mood predictions for "
        f"{sum(p is not None for p in predicted_moods)} tracks without changing observed clusters"
    )
    
    return df


def run_update():
    print("=" * 60)
    print("Starting Update Pipeline")
    print("=" * 60)
    
    try:
        # Step 1: Ingest new tracks, get metadata, get features
        print("\n[1/6] Ingesting new Spotify data (metadata and features)")
        ingestion = ingest_with_metadata()

        if not ingestion.has_new_tracks or not ingestion.ingestion_file:
            print("No new tracks found. Exiting successfully.")
            return

        latest_ingestion_file = ingestion.ingestion_file
        
        # Step 2: Assign clusters to tracks in ingestion file
        print("\n[2/6] Assigning new tracks to existing mood clusters")
        try:
            assign_clusters_to_ingestion_file(latest_ingestion_file)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print("Cluster metadata not found. Run full retrain first to initialize clusters.")
            return
        
        # Step 3: Predict mood clusters for new tracks
        print("\n[3/5] Predicting mood clusters for new tracks")
        try:
            predict_mood_for_tracks(latest_ingestion_file)
        except Exception as e:
            print(f"Warning: Could not predict mood: {e}")
            print("Continuing without mood predictions")
        
        # Step 4: Update history.parquet with no null values
        print("\n[4/5] Updating history.parquet (removing null values)")
        update_history_from_ingestion(latest_ingestion_file)
        
        # Get total track count from history.parquet (the training dataset)
        # This represents the number of tracks the models were trained on
        history_path = "data/history.parquet"
        if os.path.exists(history_path):
            history_df = pd.read_parquet(history_path)
            num_tracks = len(history_df)
            num_sessions = history_df["session_id"].nunique() if "session_id" in history_df.columns else 0
        else:
            num_tracks = 0
            num_sessions = 0
        
        # Step 5: Build and export dashboard data
        print("\n[5/6] Building and exporting dashboard data")
        try:
            build_dashboard_json()
        except Exception as e:
            print(f"Warning: Could not build dashboard: {e}")
            print("Continuing without dashboard update")
        
        # Step 6: Log metrics (track counts, drift detection)
        print("\n[6/6] Detecting drift and logging metrics")
        drift_data = None
        try:
            from models.detect_drift import detect_drift
            drift_data = detect_drift(history_path)
        except Exception as e:
            print(f"Warning: Could not detect drift: {e}")
        
        # Log metrics for update pipeline (no model metrics since we don't train)
        # num_tracks represents the total training dataset size
        log_metrics(
            num_tracks=num_tracks,
            num_sessions=num_sessions,
            mood_model_metrics=None,  # No training in update pipeline
            session_model_metrics=None,  # No training in update pipeline
            drift_data=drift_data
        )
        
        print("\n" + "=" * 60)
        print(f"Update Pipeline Completed Successfully")
        print(f"Total tracks in training dataset: {num_tracks} tracks from {num_sessions} sessions")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError in pipeline: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_update()
