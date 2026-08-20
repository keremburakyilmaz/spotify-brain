import pandas as pd
import numpy as np
import os
import pickle
import hashlib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, top_k_accuracy_score
from sklearn.preprocessing import label_binarize
import xgboost as xgb
from typing import Dict


def train_mood_model(dataset_path: str = "data/processed/mood_nexttrack_train.parquet",
                    model_path: str = "models/mood_classifier.pkl",
                    test_size: float = 0.2,
                    random_state: int = 42,
                    use_time_split: bool = True) -> Dict:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    df = pd.read_parquet(dataset_path)
    
    if df.empty:
        raise ValueError("Dataset is empty")
    
    print(f"Training mood classifier on {len(df)} samples")
    
    target_col = "target_mood_cluster"
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    metadata_cols = {target_col, "date", "target_played_at", "session_id"}
    exclude_cols = [col for col in metadata_cols if col in df.columns]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Calculate number of classes from full dataset (before splitting)
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)
    print(f"Number of classes: {n_classes} (classes: {unique_classes.tolist()})")
    
    if pd.isna(X).any():
        print("Warning: Missing values in features, filling with 0")
        X = np.nan_to_num(X, nan=0.0)
    
    if use_time_split:
        required_metadata = {"target_played_at", "session_id"}
        missing_metadata = required_metadata - set(df.columns)
        if missing_metadata:
            raise ValueError(
                "Leakage-safe mood validation requires dataset columns: "
                + ", ".join(sorted(missing_metadata))
            )

        df = df.copy()
        df["target_played_at"] = pd.to_datetime(df["target_played_at"], utc=True)
        session_order = (
            df.groupby("session_id")["target_played_at"]
            .max()
            .sort_values()
            .index.tolist()
        )
        if len(session_order) < 2:
            raise ValueError("At least two sessions are required for temporal validation")
        n_val_sessions = max(1, int(np.ceil(len(session_order) * test_size)))
        n_val_sessions = min(n_val_sessions, len(session_order) - 1)
        val_sessions = set(session_order[-n_val_sessions:])
        train_df = df[~df["session_id"].isin(val_sessions)].copy()
        val_df = df[df["session_id"].isin(val_sessions)].copy()
    else:
        from collections import Counter
        class_counts = Counter(y)
        min_class_count = min(class_counts.values())
        can_stratify = min_class_count >= 2
        stratify = df[target_col] if can_stratify else None
        train_df, val_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

    X_train = np.nan_to_num(train_df[feature_cols].values, nan=0.0)
    y_train = train_df[target_col].values
    X_val = np.nan_to_num(val_df[feature_cols].values, nan=0.0)
    y_val = val_df[target_col].values

    train_classes = np.unique(y_train)
    val_classes = np.unique(y_val)
    missing_in_train = set(unique_classes) - set(train_classes)
    if missing_in_train:
        raise ValueError(
            "Temporal training window is missing mood classes: "
            + ", ".join(str(value) for value in sorted(missing_in_train))
        )
    
    print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
    print(f"Train classes: {sorted(np.unique(y_train))}, Val classes: {sorted(np.unique(y_val))}")
    
    print("Training XGBoost classifier")
    
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        random_state=random_state,
        eval_metric='mlogloss',
        objective='multi:softprob',  # Explicitly set multi-class objective
        num_class=n_classes,  # Explicitly set number of classes
        early_stopping_rounds=20,
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Get prediction probabilities for ROC-AUC
    y_train_proba = model.predict_proba(X_train)
    y_val_proba = model.predict_proba(X_val)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    train_f1 = f1_score(y_train, y_train_pred, average='macro')
    val_f1 = f1_score(y_val, y_val_pred, average='macro')

    majority_class = int(pd.Series(y_train).mode().iloc[0])
    majority_val_accuracy = accuracy_score(
        y_val, np.full(len(y_val), majority_class)
    )
    sequence_columns = sorted(
        (col for col in feature_cols if col.startswith("mood_cluster_")),
        key=lambda col: int(col.rsplit("_", 1)[1]),
    )
    persistence_val_accuracy = None
    if sequence_columns:
        persistence_val_accuracy = accuracy_score(
            y_val, val_df[sequence_columns[-1]].values
        )
    strongest_baseline = max(
        value
        for value in [majority_val_accuracy, persistence_val_accuracy]
        if value is not None
    )
    top_k = min(3, max(1, n_classes - 1))
    val_top_k_accuracy = top_k_accuracy_score(
        y_val,
        y_val_proba,
        k=top_k,
        labels=model.classes_,
    )
    
    # ROC-AUC for multi-class (one-vs-rest)
    # Use model's classes_ to ensure consistent class ordering
    model_classes = model.classes_
    n_classes = len(model_classes)
    
    train_roc_auc = 0.0
    val_roc_auc = 0.0
    
    if n_classes >= 2:
        try:
            # Binarize labels using model's class order
            y_train_bin = label_binarize(y_train, classes=model_classes)
            y_val_bin = label_binarize(y_val, classes=model_classes)
            
            # predict_proba already returns probabilities in model.classes_ order
            # So y_train_proba and y_val_proba should match the binarized labels
            
            if n_classes == 2:
                # Binary classification
                train_roc_auc = roc_auc_score(y_train_bin[:, 1], y_train_proba[:, 1])
                val_roc_auc = roc_auc_score(y_val_bin[:, 1], y_val_proba[:, 1])
            else:
                # Multi-class: one-vs-rest with macro averaging
                train_roc_auc = roc_auc_score(y_train_bin, y_train_proba, multi_class='ovr', average='macro')
                val_roc_auc = roc_auc_score(y_val_bin, y_val_proba, multi_class='ovr', average='macro')
        except (ValueError, IndexError) as e:
            # If there's an issue (e.g., shape mismatch, missing classes), set to 0
            print(f"Warning: Could not calculate ROC-AUC: {e}")
            train_roc_auc = 0.0
            val_roc_auc = 0.0
    else:
        print("Warning: Not enough classes for ROC-AUC calculation (need at least 2)")
    
    metrics = {
        "train_accuracy": float(train_accuracy),
        "val_accuracy": float(val_accuracy),
        "train_f1_macro": float(train_f1),
        "val_f1_macro": float(val_f1),
        "train_roc_auc": float(train_roc_auc),
        "val_roc_auc": float(val_roc_auc),
        "val_top_k_accuracy": float(val_top_k_accuracy),
        "val_top_k_k": int(top_k),
        "majority_val_accuracy": float(majority_val_accuracy),
        "persistence_val_accuracy": (
            float(persistence_val_accuracy)
            if persistence_val_accuracy is not None
            else None
        ),
        "model_beats_baseline": bool(val_accuracy > strongest_baseline),
        "n_features": len(feature_cols),
        "n_train": len(X_train),
        "n_val": len(X_val)
    }
    
    print(f"Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")
    print(f"Train F1 (macro): {train_f1:.4f}, Val F1 (macro): {val_f1:.4f}")
    print(f"Train ROC-AUC: {train_roc_auc:.4f}, Val ROC-AUC: {val_roc_auc:.4f}")
    print(
        f"Val Top-{top_k}: {val_top_k_accuracy:.4f}, "
        f"Strongest baseline accuracy: {strongest_baseline:.4f}, "
        f"Model beats baseline: {val_accuracy > strongest_baseline}"
    )
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    training_timestamp = datetime.utcnow()
    data_hash_input = (
        f"{train_df['target_played_at'].min() if 'target_played_at' in train_df else None}_"
        f"{train_df['target_played_at'].max() if 'target_played_at' in train_df else None}_"
        f"{len(train_df)}"
    )
    data_hash = hashlib.md5(data_hash_input.encode()).hexdigest()[:8]
    version_hash = hashlib.md5(
        f"{training_timestamp.isoformat()}_{data_hash}".encode()
    ).hexdigest()[:8]
    metadata = {
        "training_date": training_timestamp.isoformat(),
        "data_hash": data_hash,
        "version_hash": version_hash,
        "model_beats_baseline": metrics["model_beats_baseline"],
        "fallback_strategy": (
            "persistence"
            if persistence_val_accuracy is not None
            and persistence_val_accuracy > majority_val_accuracy
            else "majority"
        ),
        "fallback_cluster": majority_class,
        "fallback_validation_accuracy": float(strongest_baseline),
        "class_priors": {
            str(int(cluster)): float((y_train == cluster).mean())
            for cluster in model.classes_
        },
    }

    with open(model_path, 'wb') as f:
        pickle.dump({
            "model": model,
            "feature_cols": feature_cols,
            "metadata": metadata,
        }, f)
    
    print(f"Saved model to {model_path}")
    
    return metrics

if __name__ == "__main__":
    train_mood_model()

