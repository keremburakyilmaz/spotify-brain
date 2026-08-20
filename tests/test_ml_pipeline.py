import os
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.export.session_predictions import build_historical_hour_probabilities
from src.export.mood_predictions import select_probability_vector
from src.features.build_mood_dataset import build_mood_dataset
from src.features.build_mood_clusters import align_cluster_identities
from src.features.mood_prediction_features import build_mood_features
from src.ingestion.spotify_ingest import ingest_with_metadata, update_history_from_ingestion
from src.models.contextual_transition import (
    fit_contextual_transition,
    predict_contextual_transition,
    recency_weights,
)
from src.models.prediction_evaluation import (
    apply_temperature,
    combine_switch_and_mood_probabilities,
    describe_predictability,
    probability_metrics,
)
from src.pipelines.run_update import predict_mood_for_tracks
from src.utils.listening_time import to_listening_time
from src.utils.sanitize_json import sanitize_value


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
    def test_cluster_alignment_preserves_previous_semantic_ids(self):
        previous = np.array([[0.1, 0.2], [0.8, 0.9]])
        reordered = np.array([[0.79, 0.88], [0.11, 0.22]])
        aligned, mapping, drift = align_cluster_identities(reordered, previous)

        self.assertEqual(mapping, {0: 1, 1: 0})
        np.testing.assert_allclose(aligned[0], reordered[1])
        self.assertLess(drift, 0.05)

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

            dataset = build_mood_dataset(history_path, output_path, window_size=3)

        self.assertIn("target_played_at", dataset.columns)
        self.assertIn("session_id", dataset.columns)
        self.assertNotIn("session_length", dataset.columns)
        self.assertIn("target_mood_switch", dataset.columns)
        self.assertIn("target_energy_direction", dataset.columns)
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

    def test_sequence_features_capture_transitions_trends_and_optional_signals(self):
        tracks = make_tracks(clusters=(0, 0, 1, 1, 2))
        tracks["skipped"] = [False, False, True, False, True]
        tracks["ms_played"] = [100, 90, 20, 80, 10]
        tracks["duration_ms"] = 100
        features = build_mood_features(
            tracks,
            session_position=5,
            session_start_time=tracks.iloc[0]["played_at"],
        )

        self.assertEqual(features["mood_transition_count"], 2)
        self.assertGreater(features["energy_trend"], 0)
        self.assertEqual(features["skip_rate_available"], 1)
        self.assertAlmostEqual(features["skip_rate"], 0.4)
        self.assertAlmostEqual(features["completion_rate_mean"], 0.6)


class ProbabilitySelectionTests(unittest.TestCase):
    def test_contextual_transition_uses_specific_sequence_before_global(self):
        frame = pd.DataFrame(
            {
                "mood_cluster_0": [0, 0, 1, 1],
                "mood_cluster_1": [1, 1, 0, 0],
                "mood_cluster_2": [1, 1, 0, 0],
                "hour_sin": [0.0] * 4,
                "hour_cos": [1.0] * 4,
                "session_position": [3] * 4,
                "target_mood_cluster": [2, 2, 1, 1],
                "target_played_at": pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC"),
            }
        )
        weights = recency_weights(frame["target_played_at"], half_life_days=90)
        artifact = fit_contextual_transition(
            frame, "target_mood_cluster", np.array([0, 1, 2]), weights, min_context_weight=1
        )
        probabilities, levels = predict_contextual_transition(frame.iloc[:1], artifact)

        self.assertEqual(levels[0], "seq3")
        self.assertEqual(int(probabilities[0].argmax()), 2)
        self.assertAlmostEqual(float(probabilities[0].sum()), 1.0)

    def test_probability_evaluation_and_abstention_are_probability_aware(self):
        classes = np.array([0, 1, 2])
        probabilities = np.array([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2]])
        metrics = probability_metrics(np.array([0, 1]), probabilities, classes)
        softened = apply_temperature(probabilities, 2.0)
        predictability = describe_predictability(
            np.array([0.34, 0.33, 0.33]),
            {"min_confidence": 0.5, "max_normalized_entropy": 0.85},
        )

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["top_2_accuracy"], 1.0)
        self.assertLess(float(softened.max()), float(probabilities.max()))
        self.assertTrue(predictability["abstained"])
        self.assertEqual(predictability["level"], "low")

    def test_two_stage_composition_reserves_stay_probability_for_current_mood(self):
        frame = pd.DataFrame({"mood_cluster_0": [0], "mood_cluster_1": [1]})
        combined = combine_switch_and_mood_probabilities(
            frame,
            np.array([0.25]),
            np.array([[0.7, 0.2, 0.1]]),
            np.array([0, 1, 2]),
        )

        self.assertAlmostEqual(combined[0, 1], 0.75)
        self.assertAlmostEqual(combined[0, [0, 2]].sum(), 0.25)

    def test_quality_gate_can_select_contextual_transition_probabilities(self):
        features = {
            "mood_cluster_0": 0,
            "mood_cluster_1": 1,
            "mood_cluster_2": 1,
            "hour_sin": 0.0,
            "hour_cos": 1.0,
            "session_position": 3,
        }
        training = pd.DataFrame(
            [
                {**features, "target_mood_cluster": 1},
                {**features, "target_mood_cluster": 1},
            ]
        )
        artifact = fit_contextual_transition(
            training,
            "target_mood_cluster",
            np.array([0, 1]),
            np.ones(2),
            min_context_weight=1,
        )
        classes, probabilities, strategy, level = select_probability_vector(
            PredictClusterZeroModel(),
            {
                "selected_strategy": "contextual_transition",
                "transition_baseline": artifact,
                "class_priors": {"0": 0.5, "1": 0.5},
            },
            features,
            np.zeros((1, len(features))),
        )

        self.assertEqual(strategy, "contextual_transition")
        self.assertEqual(level, "seq3")
        self.assertEqual(int(classes[int(probabilities.argmax())]), 1)


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

    def test_json_sanitizer_preserves_real_nulls_and_removes_nonfinite_numbers(self):
        result = sanitize_value(
            {"missing": None, "invalid": float("nan"), "legacy": "NaN"}
        )
        self.assertIsNone(result["missing"])
        self.assertIsNone(result["invalid"])
        self.assertIsNone(result["legacy"])


if __name__ == "__main__":
    unittest.main()
