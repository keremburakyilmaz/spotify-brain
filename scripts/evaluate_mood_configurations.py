#!/usr/bin/env python3
"""Compare mood window and recency settings without changing production artifacts."""

import argparse
import json
import os
import sys
import tempfile

REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPOSITORY_ROOT)

from src.features.build_mood_dataset import build_mood_dataset
from src.models.train_mood_model import train_mood_model


def parse_numbers(value, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", default="data/history.parquet")
    parser.add_argument("--windows", default="3,5,8")
    parser.add_argument("--half-lives", default="30,90,180")
    parser.add_argument("--output")
    args = parser.parse_args()

    results = []
    with tempfile.TemporaryDirectory(prefix="spotify-brain-grid-") as directory:
        for window in parse_numbers(args.windows, int):
            dataset_path = os.path.join(directory, f"mood_window_{window}.parquet")
            build_mood_dataset(args.history, dataset_path, window_size=window)
            for half_life in parse_numbers(args.half_lives, float):
                model_path = os.path.join(
                    directory, f"mood_w{window}_h{int(half_life)}.pkl"
                )
                metrics = train_mood_model(
                    dataset_path,
                    model_path,
                    recency_half_life_days=half_life,
                )
                results.append(
                    {
                        "window": window,
                        "recency_half_life_days": half_life,
                        "selected_strategy": metrics["selected_strategy"],
                        "selected_val_accuracy": metrics["selected_val_accuracy"],
                        "selected_val_log_loss": metrics["selected_val_log_loss"],
                        "candidate_metrics": metrics["candidate_metrics"],
                        "rolling_backtest": metrics["rolling_backtest"],
                    }
                )

    payload = {"configurations": results}
    rendered = json.dumps(payload, indent=2)
    if args.output:
        with open(args.output, "w") as output_file:
            output_file.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
