import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
from typing import Dict


def train_session_model(dataset_path: str = "data/processed/session_start_train.parquet",
                       model_path: str = "models/session_classifier.pkl",
                       test_size: float = 0.2,
                       random_state: int = 42,
                       use_time_split: bool = True) -> Dict:
    # Load dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    df = pd.read_parquet(dataset_path)
    
    if df.empty:
        raise ValueError("Dataset is empty")
    
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
    if use_time_split and "date" in df.columns:
        # Time-based split: use earlier data for training
        df_sorted = df.sort_values("date").reset_index(drop=True)
        split_idx = int(len(df_sorted) * (1 - test_size))
        
        X_train = df_sorted.loc[:split_idx, feature_cols].values
        y_train = df_sorted.loc[:split_idx, target_col].values
        X_val = df_sorted.loc[split_idx:, feature_cols].values
        y_val = df_sorted.loc[split_idx:, target_col].values
        
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
    
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump({
            "model": model,
            "feature_cols": feature_cols
        }, f)
    
    print(f"Saved model to {model_path}")
    
    return metrics

if __name__ == "__main__":
    train_session_model()






