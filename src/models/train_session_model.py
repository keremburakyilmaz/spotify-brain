import pandas as pd
import numpy as np
import os
import pickle
import hashlib
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
from typing import Dict, Optional
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.evaluate_session_model import evaluate_session_model

def train_session_model(dataset_path: str = "data/processed/session_start_train.parquet",
                       model_path: str = "models/session_classifier.pkl",
                       test_size: float = 0.2,
                       random_state: int = 42,
                       use_time_split: bool = True,
                       rolling_window_days: Optional[int] = None) -> Dict:
    # Load dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    df = pd.read_parquet(dataset_path)
    
    if df.empty:
        raise ValueError("Dataset is empty")
    
    # Apply rolling window if specified
    if rolling_window_days is not None and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        max_date = df["date"].max()
        window_start = max_date - timedelta(days=rolling_window_days)
        df = df[df["date"] >= window_start].copy()
        print(f"Applied rolling window: using last {rolling_window_days} days")
        print(f"Training window: {window_start.date()} to {max_date.date()}")
    
    if df.empty:
        raise ValueError("Dataset is empty after applying rolling window")
    
    print(f"Training session classifier on {len(df)} samples")
    
    # Separate features and target
    target_col = "target_session_start"
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    # Get feature columns (exclude target and any metadata)
    exclude_cols = [target_col, "date"] if "date" in df.columns else [target_col]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Handle missing values
    if pd.isna(X).any():
        print("Warning: Missing values in features, filling with 0")
        X = np.nan_to_num(X, nan=0.0)
    
    # Split data
    train_start_date = None
    train_end_date = None
    val_start_date = None
    val_end_date = None
    val_df = None  # Initialize for use in evaluation
    
    if use_time_split and "date" in df.columns:
        # Time-based split: use earlier data for training
        df_sorted = df.sort_values("date").reset_index(drop=True)
        split_idx = int(len(df_sorted) * (1 - test_size))
        
        train_df = df_sorted.loc[:split_idx].copy()
        val_df = df_sorted.loc[split_idx:].copy()
        
        # Record date ranges for metadata
        train_start_date = train_df["date"].min()
        train_end_date = train_df["date"].max()
        val_start_date = val_df["date"].min()
        val_end_date = val_df["date"].max()
        
        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_val = val_df[feature_cols].values
        y_val = val_df[target_col].values
        
        # Handle missing values
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0)
    else:
        # Random split - check if stratification is possible
        from collections import Counter
        class_counts = Counter(y)
        min_class_count = min(class_counts.values())
        can_stratify = min_class_count >= 2
        
        if can_stratify:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            print(f"Warning: Cannot use stratified split (min class count: {min_class_count}). Using random split instead.")
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        val_df = df.tail(len(y_val)).copy()
    
    print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
    print(f"Positive samples - Train: {y_train.sum()}, Val: {y_val.sum()}")
    
    # Train XGBoost classifier
    print("Training XGBoost classifier")
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state,
        eval_metric='logloss'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Evaluate
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    # ROC-AUC
    try:
        train_roc_auc = roc_auc_score(y_train, y_train_proba)
        val_roc_auc = roc_auc_score(y_val, y_val_proba)
    except ValueError:
        # If only one class in validation set
        train_roc_auc = 0.0
        val_roc_auc = 0.0
    
    metrics = {
        "train_accuracy": float(train_accuracy),
        "val_accuracy": float(val_accuracy),
        "train_roc_auc": float(train_roc_auc),
        "val_roc_auc": float(val_roc_auc),
        "n_features": len(feature_cols),
        "n_train": len(X_train),
        "n_val": len(X_val)
    }
    
    print(f"Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")
    print(f"Train ROC-AUC: {train_roc_auc:.4f}, Val ROC-AUC: {val_roc_auc:.4f}")
    
    # Comprehensive evaluation with baselines
    print("\nRunning comprehensive evaluation with baselines...")
    
    # Prepare validation dataframe for evaluation
    if val_df is not None:
        val_df_for_eval = val_df.copy()
    else:
        # For random split, create a minimal dataframe with hour column if available
        val_df_for_eval = pd.DataFrame()
        if "hour" in df.columns:
            # Approximate: use last N rows matching validation set size
            val_df_for_eval["hour"] = df["hour"].tail(len(y_val)).values
        if "date" in df.columns:
            val_df_for_eval["date"] = df["date"].tail(len(y_val)).values
    
    # Only run evaluation if we have the necessary data
    try:
        if val_df_for_eval is not None and len(val_df_for_eval) > 0:
            evaluation_results = evaluate_session_model(
                val_df_for_eval,
                y_val,
                y_val_pred,
                y_val_proba,
                date_col="date" if "date" in val_df_for_eval.columns else None,
                hour_col="hour" if "hour" in val_df_for_eval.columns else None,
                target_col=target_col
            )
        else:
            # Fallback: use original df with approximate indices
            evaluation_results = evaluate_session_model(
                df.tail(len(y_val)),
                y_val,
                y_val_pred,
                y_val_proba,
                date_col="date" if "date" in df.columns else None,
                hour_col="hour" if "hour" in df.columns else None,
                target_col=target_col
            )
    except Exception as e:
        print(f"Warning: Could not run comprehensive evaluation: {e}")
        evaluation_results = {}
    
    # Add evaluation results to metrics
    metrics["evaluation"] = evaluation_results
    
    # Generate model metadata
    training_timestamp = datetime.utcnow()
    
    # Create data hash for versioning
    data_hash_input = f"{train_start_date}_{train_end_date}_{len(X_train)}"
    data_hash = hashlib.md5(data_hash_input.encode()).hexdigest()[:8]
    
    # Model version hash
    version_hash_input = f"{training_timestamp.isoformat()}_{data_hash}"
    version_hash = hashlib.md5(version_hash_input.encode()).hexdigest()[:8]
    
    model_metadata = {
        "training_date": training_timestamp.isoformat(),
        "training_window_start": train_start_date.isoformat() if train_start_date is not None else None,
        "training_window_end": train_end_date.isoformat() if train_end_date is not None else None,
        "validation_window_start": val_start_date.isoformat() if val_start_date is not None else None,
        "validation_window_end": val_end_date.isoformat() if val_end_date is not None else None,
        "data_hash": data_hash,
        "version_hash": version_hash,
        "rolling_window_days": rolling_window_days
    }
    
    # Save model with metadata
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump({
            "model": model,
            "feature_cols": feature_cols,
            "metadata": model_metadata
        }, f)
    
    print(f"Saved model to {model_path}")
    print(f"Model version: {version_hash}")
    print(f"Training window: {train_start_date if train_start_date else 'N/A'} to {train_end_date if train_end_date else 'N/A'}")
    
    metrics["model_metadata"] = model_metadata
    
    return metrics

if __name__ == "__main__":
    train_session_model()






