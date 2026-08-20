import hashlib
import os
import pickle
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

try:
    from models.contextual_transition import (
        fit_contextual_transition,
        persistence_probabilities,
        predict_contextual_transition,
        recency_weights,
    )
    from models.prediction_evaluation import (
        apply_temperature,
        combine_switch_and_mood_probabilities,
        derive_abstention_policy,
        fit_temperature,
        probability_metrics,
    )
except ModuleNotFoundError:
    from .contextual_transition import (
        fit_contextual_transition,
        persistence_probabilities,
        predict_contextual_transition,
        recency_weights,
    )
    from .prediction_evaluation import (
        apply_temperature,
        combine_switch_and_mood_probabilities,
        derive_abstention_policy,
        fit_temperature,
        probability_metrics,
    )


def _temporal_session_split(df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    session_order = (
        df.groupby("session_id")["target_played_at"].max().sort_values().index.tolist()
    )
    if len(session_order) < 2:
        raise ValueError("At least two sessions are required for temporal validation")
    n_validation = min(
        max(1, int(np.ceil(len(session_order) * test_size))),
        len(session_order) - 1,
    )
    validation_sessions = set(session_order[-n_validation:])
    return (
        df[~df["session_id"].isin(validation_sessions)].copy(),
        df[df["session_id"].isin(validation_sessions)].copy(),
    )


def _inner_calibration_split(train_df: pd.DataFrame, classes: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ordered_sessions = (
        train_df.groupby("session_id")["target_played_at"].max().sort_values().index.tolist()
    )
    if len(ordered_sessions) < 3:
        return train_df, train_df.iloc[0:0].copy()
    n_calibration = max(1, int(np.ceil(len(ordered_sessions) * 0.15)))
    calibration_sessions = set(ordered_sessions[-n_calibration:])
    fit_df = train_df[~train_df["session_id"].isin(calibration_sessions)].copy()
    calibration_df = train_df[train_df["session_id"].isin(calibration_sessions)].copy()
    if set(classes) - set(fit_df["target_mood_cluster"].unique()):
        return train_df, train_df.iloc[0:0].copy()
    return fit_df, calibration_df


def _roc_auc(y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> float:
    try:
        encoded = label_binarize(y_true, classes=classes)
        if len(classes) == 2:
            return float(roc_auc_score(encoded[:, 0], probabilities[:, 1]))
        return float(roc_auc_score(encoded, probabilities, multi_class="ovr", average="macro"))
    except (ValueError, IndexError):
        return 0.0


def _selection_score(metrics: Dict[str, float]) -> float:
    return float(metrics["log_loss"] + 0.25 * (1.0 - metrics["accuracy"]))


def _switch_metrics(frame: pd.DataFrame, y_true: np.ndarray,
                    probabilities: np.ndarray, classes: np.ndarray) -> Dict[str, float]:
    sequence_columns = sorted(
        (column for column in frame.columns if column.startswith("mood_cluster_")),
        key=lambda column: int(column.rsplit("_", 1)[1]),
    )
    if not sequence_columns:
        return {}
    previous = frame[sequence_columns[-1]].to_numpy(dtype=int)
    class_indices = {int(value): index for index, value in enumerate(classes)}
    stay_probability = np.array([
        probabilities[index, class_indices[value]]
        for index, value in enumerate(previous)
    ])
    switch_probability = 1.0 - stay_probability
    switched = (y_true != previous).astype(int)
    output = {
        "switch_accuracy": float(
            np.mean((switch_probability >= 0.5).astype(int) == switched)
        ),
        "switch_brier_score": float(np.mean((switch_probability - switched) ** 2)),
    }
    if len(np.unique(switched)) == 2:
        output["switch_roc_auc"] = float(roc_auc_score(switched, switch_probability))
    return output


def _fit_switch_model(
    fit_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    feature_cols,
    random_state: int,
    recency_half_life_days: float,
) -> Tuple[object, float]:
    y_fit = fit_df["target_mood_switch"].to_numpy(dtype=int)
    positives = max(1, int(y_fit.sum()))
    negatives = max(1, int(len(y_fit) - y_fit.sum()))
    parameters = {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.04,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 3.0,
        "reg_alpha": 0.2,
        "scale_pos_weight": negatives / positives,
        "random_state": random_state,
        "eval_metric": "logloss",
        "objective": "binary:logistic",
    }
    can_calibrate = (
        not calibration_df.empty
        and calibration_df["target_mood_switch"].nunique() == 2
    )
    if can_calibrate:
        parameters["early_stopping_rounds"] = 25
    switch_model = xgb.XGBClassifier(**parameters)
    fit_kwargs = {
        "sample_weight": recency_weights(
            fit_df["target_played_at"], recency_half_life_days
        ),
        "verbose": False,
    }
    if can_calibrate:
        X_calibration = np.nan_to_num(
            calibration_df[feature_cols].to_numpy(dtype=float), nan=0.0
        )
        y_calibration = calibration_df["target_mood_switch"].to_numpy(dtype=int)
        fit_kwargs.update(
            {
                "eval_set": [(X_calibration, y_calibration)],
                "sample_weight_eval_set": [
                    recency_weights(
                        calibration_df["target_played_at"], recency_half_life_days
                    )
                ],
            }
        )
    switch_model.fit(
        np.nan_to_num(fit_df[feature_cols].to_numpy(dtype=float), nan=0.0),
        y_fit,
        **fit_kwargs,
    )
    temperature = 1.0
    if can_calibrate:
        temperature = fit_temperature(
            y_calibration,
            switch_model.predict_proba(X_calibration),
            np.array([0, 1]),
        )
    return switch_model, temperature


def _rolling_session_folds(df: pd.DataFrame, n_folds: int = 3):
    sessions = (
        df.groupby("session_id")["target_played_at"].max().sort_values().index.tolist()
    )
    if len(sessions) < 8:
        return []
    initial_train = max(2, len(sessions) // 2)
    remaining = len(sessions) - initial_train
    fold_size = max(1, remaining // n_folds)
    folds = []
    for fold_index in range(n_folds):
        validation_start = initial_train + fold_index * fold_size
        validation_end = (
            len(sessions)
            if fold_index == n_folds - 1
            else min(len(sessions), validation_start + fold_size)
        )
        if validation_start >= len(sessions):
            break
        train_sessions = set(sessions[:validation_start])
        validation_sessions = set(sessions[validation_start:validation_end])
        folds.append(
            (
                df[df["session_id"].isin(train_sessions)].copy(),
                df[df["session_id"].isin(validation_sessions)].copy(),
            )
        )
    return folds


def _rolling_backtest(
    df: pd.DataFrame,
    feature_cols,
    classes: np.ndarray,
    random_state: int,
    recency_half_life_days: float,
) -> Dict:
    fold_results = []
    for fold_index, (fold_train, fold_val) in enumerate(_rolling_session_folds(df)):
        if set(classes) - set(fold_train["target_mood_cluster"].unique()):
            continue
        fit_df, calibration_df = _inner_calibration_split(fold_train, classes)
        parameters = {
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.04,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 3.0,
            "reg_alpha": 0.2,
            "random_state": random_state + fold_index,
            "eval_metric": "mlogloss",
            "objective": "multi:softprob",
            "num_class": len(classes),
        }
        can_calibrate = not calibration_df.empty
        if can_calibrate:
            parameters["early_stopping_rounds"] = 20
        mood_model = xgb.XGBClassifier(**parameters)
        fit_kwargs = {
            "sample_weight": recency_weights(
                fit_df["target_played_at"], recency_half_life_days
            ),
            "verbose": False,
        }
        if can_calibrate:
            X_calibration = np.nan_to_num(
                calibration_df[feature_cols].to_numpy(dtype=float), nan=0.0
            )
            y_calibration = calibration_df["target_mood_cluster"].to_numpy(dtype=int)
            fit_kwargs.update(
                {
                    "eval_set": [(X_calibration, y_calibration)],
                    "sample_weight_eval_set": [
                        recency_weights(
                            calibration_df["target_played_at"], recency_half_life_days
                        )
                    ],
                }
            )
        mood_model.fit(
            np.nan_to_num(fit_df[feature_cols].to_numpy(dtype=float), nan=0.0),
            fit_df["target_mood_cluster"].to_numpy(dtype=int),
            **fit_kwargs,
        )
        mood_temperature = (
            fit_temperature(
                y_calibration,
                mood_model.predict_proba(X_calibration),
                classes,
            )
            if can_calibrate
            else 1.0
        )
        switch_model, switch_temperature = _fit_switch_model(
            fit_df,
            calibration_df,
            feature_cols,
            random_state + fold_index,
            recency_half_life_days,
        )
        X_val = np.nan_to_num(fold_val[feature_cols].to_numpy(dtype=float), nan=0.0)
        y_val = fold_val["target_mood_cluster"].to_numpy(dtype=int)
        mood_probabilities = apply_temperature(
            mood_model.predict_proba(X_val), mood_temperature
        )
        switch_probabilities = apply_temperature(
            switch_model.predict_proba(X_val), switch_temperature
        )[:, 1]
        two_stage = combine_switch_and_mood_probabilities(
            fold_val, switch_probabilities, mood_probabilities, classes
        )
        weights = recency_weights(fold_train["target_played_at"], recency_half_life_days)
        y_train = fold_train["target_mood_cluster"].to_numpy(dtype=int)
        counts = np.array([weights[y_train == value].sum() for value in classes])
        priors = counts / counts.sum()
        transition = fit_contextual_transition(
            fold_train, "target_mood_cluster", classes, weights
        )
        transition_probabilities, _ = predict_contextual_transition(fold_val, transition)
        candidates = {
            "model": mood_probabilities,
            "two_stage_model": two_stage,
            "contextual_transition": transition_probabilities,
            "majority": np.tile(priors, (len(fold_val), 1)),
            "persistence": persistence_probabilities(
                fold_val, classes, priors, 0.85
            ),
        }
        fold_metrics = {}
        for name, probabilities in candidates.items():
            values = probability_metrics(y_val, probabilities, classes)
            values["selection_score"] = _selection_score(values)
            fold_metrics[name] = values
        fold_results.append(fold_metrics)

    if not fold_results:
        return {"n_folds": 0, "candidates": {}}
    aggregate = {}
    for candidate in fold_results[0]:
        aggregate[candidate] = {}
        for metric in ["accuracy", "log_loss", "brier_score", "top_2_accuracy", "selection_score"]:
            values = [fold[candidate][metric] for fold in fold_results]
            aggregate[candidate][f"mean_{metric}"] = float(np.mean(values))
            aggregate[candidate][f"std_{metric}"] = float(np.std(values))
    return {"n_folds": len(fold_results), "candidates": aggregate}


def train_mood_model(
    dataset_path: str = "data/processed/mood_nexttrack_train.parquet",
    model_path: str = "models/mood_classifier.pkl",
    test_size: float = 0.2,
    random_state: int = 42,
    use_time_split: bool = True,
    recency_half_life_days: float = 90.0,
) -> Dict:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    if df.empty:
        raise ValueError("Dataset is empty")

    target_col = "target_mood_cluster"
    if target_col not in df:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    df = df.copy()
    df["target_played_at"] = pd.to_datetime(df["target_played_at"], utc=True)
    metadata_cols = {
        column
        for column in df.columns
        if column.startswith("target_") or column in {"date", "session_id"}
    }
    feature_cols = [column for column in df.columns if column not in metadata_cols]
    classes = np.sort(df[target_col].unique().astype(int))
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError("Mood cluster IDs must be contiguous and start at zero")

    if use_time_split:
        required = {"target_played_at", "session_id"}
        if required - set(df.columns):
            raise ValueError("Leakage-safe validation requires target_played_at and session_id")
        train_df, val_df = _temporal_session_split(df, test_size)
    else:
        train_df, val_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[target_col] if df[target_col].value_counts().min() >= 2 else None,
        )
    if set(classes) - set(train_df[target_col].unique()):
        raise ValueError("Temporal training window is missing one or more mood classes")

    fit_df, calibration_df = _inner_calibration_split(train_df, classes)
    X_fit = np.nan_to_num(fit_df[feature_cols].to_numpy(dtype=float), nan=0.0)
    y_fit = fit_df[target_col].to_numpy(dtype=int)
    fit_weights = recency_weights(fit_df["target_played_at"], recency_half_life_days)

    model_parameters = {
        "n_estimators": 400,
        "max_depth": 3,
        "learning_rate": 0.04,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 3.0,
        "reg_alpha": 0.2,
        "random_state": random_state,
        "eval_metric": "mlogloss",
        "objective": "multi:softprob",
        "num_class": len(classes),
    }
    if not calibration_df.empty:
        model_parameters["early_stopping_rounds"] = 25
    model = xgb.XGBClassifier(**model_parameters)
    fit_kwargs = {"sample_weight": fit_weights, "verbose": False}
    if not calibration_df.empty:
        X_calibration = np.nan_to_num(
            calibration_df[feature_cols].to_numpy(dtype=float), nan=0.0
        )
        y_calibration = calibration_df[target_col].to_numpy(dtype=int)
        calibration_weights = recency_weights(
            calibration_df["target_played_at"], recency_half_life_days
        )
        fit_kwargs.update(
            {
                "eval_set": [(X_calibration, y_calibration)],
                "sample_weight_eval_set": [calibration_weights],
            }
        )
    model.fit(X_fit, y_fit, **fit_kwargs)
    switch_model, switch_temperature = _fit_switch_model(
        fit_df,
        calibration_df,
        feature_cols,
        random_state,
        recency_half_life_days,
    )

    temperature = 1.0
    if not calibration_df.empty:
        temperature = fit_temperature(
            y_calibration, model.predict_proba(X_calibration), classes
        )

    X_train = np.nan_to_num(train_df[feature_cols].to_numpy(dtype=float), nan=0.0)
    X_val = np.nan_to_num(val_df[feature_cols].to_numpy(dtype=float), nan=0.0)
    y_train = train_df[target_col].to_numpy(dtype=int)
    y_val = val_df[target_col].to_numpy(dtype=int)
    model_train_probabilities = apply_temperature(model.predict_proba(X_train), temperature)
    model_val_probabilities = apply_temperature(model.predict_proba(X_val), temperature)
    switch_train_probabilities = apply_temperature(
        switch_model.predict_proba(X_train), switch_temperature
    )[:, 1]
    switch_val_probabilities = apply_temperature(
        switch_model.predict_proba(X_val), switch_temperature
    )[:, 1]
    two_stage_train_probabilities = combine_switch_and_mood_probabilities(
        train_df, switch_train_probabilities, model_train_probabilities, classes
    )
    two_stage_val_probabilities = combine_switch_and_mood_probabilities(
        val_df, switch_val_probabilities, model_val_probabilities, classes
    )

    train_weights = recency_weights(train_df["target_played_at"], recency_half_life_days)
    weighted_counts = np.array([
        train_weights[y_train == mood_class].sum() for mood_class in classes
    ])
    priors = weighted_counts / weighted_counts.sum()
    majority_probabilities = np.tile(priors, (len(val_df), 1))
    persistence_strength = 0.85
    persistence_values = persistence_probabilities(
        val_df, classes, priors, persistence_strength
    )
    transition_artifact = fit_contextual_transition(
        train_df, target_col, classes, train_weights
    )
    transition_probabilities, transition_levels = predict_contextual_transition(
        val_df, transition_artifact
    )

    candidate_probabilities = {
        "model": model_val_probabilities,
        "two_stage_model": two_stage_val_probabilities,
        "contextual_transition": transition_probabilities,
        "majority": majority_probabilities,
        "persistence": persistence_values,
    }
    candidate_metrics = {
        name: probability_metrics(y_val, values, classes)
        for name, values in candidate_probabilities.items()
    }
    for name, values in candidate_metrics.items():
        values.update(_switch_metrics(val_df, y_val, candidate_probabilities[name], classes))
        values["selection_score"] = _selection_score(values)

    rolling_backtest = _rolling_backtest(
        df, feature_cols, classes, random_state, recency_half_life_days
    )
    rolling_candidates = rolling_backtest.get("candidates", {})
    score_key = "mean_selection_score" if rolling_candidates else "selection_score"
    selection_metrics = rolling_candidates or candidate_metrics
    best_baseline = min(
        ("contextual_transition", "majority", "persistence"),
        key=lambda name: selection_metrics[name][score_key],
    )
    best_learned = min(
        ("model", "two_stage_model"),
        key=lambda name: selection_metrics[name][score_key],
    )
    outer_supports_learned = (
        candidate_metrics[best_learned]["selection_score"]
        < candidate_metrics[best_baseline]["selection_score"] * 0.995
        and candidate_metrics[best_learned]["accuracy"]
        >= candidate_metrics[best_baseline]["accuracy"] - 0.01
    )
    rolling_supports_learned = (
        not rolling_candidates
        or (
            rolling_candidates[best_learned]["mean_selection_score"]
            < rolling_candidates[best_baseline]["mean_selection_score"] * 0.995
            and rolling_candidates[best_learned]["mean_accuracy"]
            >= rolling_candidates[best_baseline]["mean_accuracy"] - 0.01
        )
    )
    model_beats_baseline = outer_supports_learned and rolling_supports_learned
    selected_strategy = best_learned if model_beats_baseline else best_baseline
    selected_probabilities = candidate_probabilities[selected_strategy]
    abstention_policy = derive_abstention_policy(y_val, selected_probabilities, classes)

    model_train_metrics = probability_metrics(y_train, model_train_probabilities, classes)
    two_stage_train_metrics = probability_metrics(
        y_train, two_stage_train_probabilities, classes
    )
    model_metrics = candidate_metrics["model"]
    selected_metrics = candidate_metrics[selected_strategy]
    metrics = {
        "train_accuracy": model_train_metrics["accuracy"],
        "val_accuracy": model_metrics["accuracy"],
        "train_f1_macro": model_train_metrics["f1_macro"],
        "val_f1_macro": model_metrics["f1_macro"],
        "train_roc_auc": _roc_auc(y_train, model_train_probabilities, classes),
        "val_roc_auc": _roc_auc(y_val, model_val_probabilities, classes),
        "val_top_k_accuracy": model_metrics.get("top_3_accuracy", model_metrics["accuracy"]),
        "val_top_k_k": min(3, len(classes)),
        "majority_val_accuracy": candidate_metrics["majority"]["accuracy"],
        "persistence_val_accuracy": candidate_metrics["persistence"]["accuracy"],
        "contextual_transition_val_accuracy": candidate_metrics["contextual_transition"]["accuracy"],
        "model_beats_baseline": bool(model_beats_baseline),
        "selected_strategy": selected_strategy,
        "selected_val_accuracy": selected_metrics["accuracy"],
        "selected_val_log_loss": selected_metrics["log_loss"],
        "temperature": float(temperature),
        "switch_temperature": float(switch_temperature),
        "two_stage_train_metrics": two_stage_train_metrics,
        "candidate_metrics": candidate_metrics,
        "rolling_backtest": rolling_backtest,
        "transition_context_usage": pd.Series(transition_levels).value_counts(normalize=True).to_dict(),
        "abstention_policy": abstention_policy,
        "n_features": len(feature_cols),
        "n_train": len(train_df),
        "n_val": len(val_df),
    }

    print(
        f"Mood candidates: model={model_metrics['accuracy']:.4f}/{model_metrics['log_loss']:.4f}, "
        f"two-stage={candidate_metrics['two_stage_model']['accuracy']:.4f}/"
        f"{candidate_metrics['two_stage_model']['log_loss']:.4f}, "
        f"transition={candidate_metrics['contextual_transition']['accuracy']:.4f}/"
        f"{candidate_metrics['contextual_transition']['log_loss']:.4f}, selected={selected_strategy}"
    )
    print(
        f"Selected top-2={selected_metrics.get('top_2_accuracy', 1.0):.4f}, "
        f"top-3={selected_metrics.get('top_3_accuracy', 1.0):.4f}, "
        f"ECE={selected_metrics['expected_calibration_error']:.4f}, "
        f"abstention coverage={abstention_policy['validation_coverage']:.4f}"
    )
    if rolling_candidates:
        print(
            f"Rolling {rolling_backtest['n_folds']}-fold selection: "
            f"learned={best_learned} "
            f"({rolling_candidates[best_learned]['mean_selection_score']:.4f}), "
            f"baseline={best_baseline} "
            f"({rolling_candidates[best_baseline]['mean_selection_score']:.4f}), "
            f"learned gate={model_beats_baseline}"
        )

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    training_timestamp = datetime.utcnow()
    data_hash_input = (
        f"{train_df['target_played_at'].min()}_{train_df['target_played_at'].max()}_"
        f"{len(train_df)}_{len(feature_cols)}"
    )
    data_hash = hashlib.md5(data_hash_input.encode()).hexdigest()[:8]
    version_hash = hashlib.md5(
        f"{training_timestamp.isoformat()}_{data_hash}".encode()
    ).hexdigest()[:8]
    metadata = {
        "training_date": training_timestamp.isoformat(),
        "data_hash": data_hash,
        "version_hash": version_hash,
        "window_size": len(
            [column for column in feature_cols if column.startswith("mood_cluster_")]
        ),
        "recency_half_life_days": float(recency_half_life_days),
        "temperature": float(temperature),
        "switch_temperature": float(switch_temperature),
        "model_beats_baseline": bool(model_beats_baseline),
        "selected_strategy": selected_strategy,
        "fallback_strategy": best_baseline,
        "fallback_cluster": int(classes[np.argmax(priors)]),
        "fallback_validation_accuracy": candidate_metrics[best_baseline]["accuracy"],
        "selected_validation_accuracy": selected_metrics["accuracy"],
        "selected_validation_log_loss": selected_metrics["log_loss"],
        "class_priors": {
            str(int(cluster)): float(probability)
            for cluster, probability in zip(classes, priors)
        },
        "transition_baseline": transition_artifact,
        "persistence_strength": persistence_strength,
        "abstention_policy": abstention_policy,
        "candidate_metrics": candidate_metrics,
        "rolling_backtest": rolling_backtest,
    }
    with open(model_path, "wb") as model_file:
        pickle.dump(
            {
                "model": model,
                "switch_model": switch_model,
                "feature_cols": feature_cols,
                "metadata": metadata,
            },
            model_file,
        )
    print(f"Saved model to {model_path}")
    return metrics


if __name__ == "__main__":
    train_mood_model()
