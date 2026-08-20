import os
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.export.session_predictions import build_historical_hour_probabilities
from src.features.build_mood_dataset import build_mood_dataset
from src.features.mood_prediction_features import build_mood_features
from src.ingestion.spotify_ingest import ingest_with_metadata, update_history_from_ingestion
from src.pipelines.run_update import predict_mood_for_tracks
from src.utils.listening_time import to_listening_time


class PredictClusterZeroModel:
    classes_ = np.array([0, 1])

    def predict(self, _features):
        return np.array([0])

    def predict_proba(self, _features):
        return np.array([[0.8, 0.2]])


def make_tracks(session_id=7, clusters=(0, 0, 1, 1)):
    timestamps = pd.date_range("2026-01-01T21:00:00Z", periods=len(clusters), freq="5min")
    rows = []
    for index, (timestamp, cluster) in enumerate(zip(timestamps, clusters)):
        rows.append(
            {
                "track_id": f"track-{session_id}-{index}",
                "track_name": f"Track {index}",
                "artist_name": "Artist",
                "played_at": timestamp,
                "session_id": session_id,
                "mood_cluster_id": cluster,
                "valence": 0.2 + index * 0.1,
                "energy": 0.4 + index * 0.05,
                "danceability": 0.5,
                "acousticness": 0.1,
                "instrumentalness": 0.0,
                "tempo": 120.0,
                "tempo_norm": 0.48,
                "image_url": None,
            }
        )
    return pd.DataFrame(rows)


class MoodDatasetTests(unittest.TestCase):
    @patch.dict(os.environ, {"LISTENING_TIMEZONE": "Europe/Istanbul"})
    def test_dataset_keeps_temporal_metadata_and_uses_observable_features(self):
        history = pd.concat(
            [make_tracks(session_id=7), make_tracks(session_id=8)],
            ignore_index=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            history_path = os.path.join(directory, "history.parquet")
            output_path = os.path.join(directory, "mood.parquet")
            history.to_parquet(history_path, index=False)

            dataset = build_mood_dataset(history_path, output_path)

        self.assertIn("target_played_at", dataset.columns)
        self.assertIn("session_id", dataset.columns)
        self.assertNotIn("session_length", dataset.columns)
        self.assertEqual(set(dataset["session_id"]), {7, 8})
        # 21:10 UTC is 00:10 the next day in Istanbul.
        self.assertAlmostEqual(dataset.iloc[0]["hour_sin"], 0.0, places=7)

        local_history = history[history["session_id"] == 7].copy()
        local_history["played_at"] = to_listening_time(local_history["played_at"])
        expected = build_mood_features(
            local_history.iloc[:3],
            session_position=3,
            session_start_time=local_history.iloc[0]["played_at"],
        )
        for feature, value in expected.items():
            self.assertAlmostEqual(dataset.iloc[0][feature], value)


class IncrementalPredictionTests(unittest.TestCase):
    @patch.dict(os.environ, {"LISTENING_TIMEZONE": "Europe/Istanbul"})
    def test_online_prediction_never_overwrites_observed_cluster(self):
        history = make_tracks(clusters=(0, 0, 0)).iloc[:3].copy()
        ingestion = make_tracks(clusters=(0, 0, 0, 1)).iloc[3:].copy()
        feature_frame = history.copy()
        feature_frame["played_at"] = to_listening_time(feature_frame["played_at"])
        feature_cols = list(
            build_mood_features(
                feature_frame,
                session_position=3,
                session_start_time=feature_frame.iloc[0]["played_at"],
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            history_path = os.path.join(directory, "history.parquet")
            ingestion_path = os.path.join(directory, "ingestion.parquet")
            model_path = os.path.join(directory, "model.pkl")
            history.to_parquet(history_path, index=False)
            ingestion.to_parquet(ingestion_path, index=False)
            with open(model_path, "wb") as model_file:
                pickle.dump(
                    {"model": PredictClusterZeroModel(), "feature_cols": feature_cols},
                    model_file,
                )

            result = predict_mood_for_tracks(
                ingestion_path,
                history_path=history_path,
                mood_model_path=model_path,
            )

        self.assertEqual(result.iloc[0]["mood_cluster_id"], 1)
        self.assertEqual(result.iloc[0]["predicted_mood_cluster_id"], 0)
        self.assertAlmostEqual(result.iloc[0]["prediction_confidence"], 0.8)


class IngestionAndSessionBaselineTests(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "SPOTIFY_CLIENT_ID": "client",
            "SPOTIFY_CLIENT_SECRET": "secret",
            "SPOTIFY_REFRESH_TOKEN": "refresh",
        },
    )
    @patch("src.ingestion.spotify_ingest.SpotifyIngester.fetch_recently_played")
    def test_no_new_tracks_does_not_reuse_an_old_ingestion(self, fetch_recently_played):
        fetch_recently_played.return_value = []
        with tempfile.TemporaryDirectory() as directory:
            history_path = os.path.join(directory, "history.parquet")
            make_tracks().to_parquet(history_path, index=False)
            result = ingest_with_metadata(history_path)

        self.assertFalse(result.has_new_tracks)
        self.assertIsNone(result.ingestion_file)

    def test_optional_prediction_nulls_do_not_drop_history_rows(self):
        existing = make_tracks(clusters=(0,)).copy()
        new_track = make_tracks(session_id=8, clusters=(1,)).copy()
        new_track["predicted_mood_cluster_id"] = pd.array([None], dtype="Int64")
        new_track["prediction_confidence"] = pd.array([None], dtype="Float64")

        with tempfile.TemporaryDirectory() as directory:
            history_path = os.path.join(directory, "history.parquet")
            ingestion_path = os.path.join(directory, "ingestion.parquet")
            existing.to_parquet(history_path, index=False)
            new_track.to_parquet(ingestion_path, index=False)
            result = update_history_from_ingestion(ingestion_path, history_path)

        self.assertEqual(len(result), 2)
        self.assertTrue(pd.isna(result.iloc[-1]["predicted_mood_cluster_id"]))

    def test_historical_hour_baseline_is_a_probability_not_a_normalized_count(self):
        history = make_tracks(session_id=1).iloc[:1].copy()
        second = make_tracks(session_id=2).iloc[:1].copy()
        second["played_at"] = second["played_at"] + pd.offsets.Day(1)
        history = pd.concat([history, second], ignore_index=True)
        probabilities = build_historical_hour_probabilities(
            history, pd.Timestamp("2026-01-05").date()
        )

        self.assertTrue(all(0.0 < probability < 1.0 for probability in probabilities.values()))
        self.assertGreater(probabilities["21"], probabilities["20"])


if __name__ == "__main__":
    unittest.main()
