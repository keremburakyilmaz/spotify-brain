import pandas as pd
import numpy as np
import os
import pickle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import xgboost as xgb
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def get_new_ingestion_files(ingestions_dir: str = "data/ingestions",
                           last_processed_file: Optional[str] = None,
                           history_path: str = "data/history.parquet") -> List[str]:
    if last_processed_file is None:
        # First time: return history.parquet
        if os.path.exists(history_path):
            return [history_path]
        return []
    
    if not os.path.exists(ingestions_dir):
        return []
    
    # Get all ingestion files
    all_files = [f for f in os.listdir(ingestions_dir) if f.startswith("ingestion_") and f.endswith(".parquet")]
    
    if not all_files:
        return []
    
    # Sort by filename (which includes timestamp)
    all_files.sort()
    
    # Find index of last processed file
    try:
        last_idx = all_files.index(os.path.basename(last_processed_file))
        # Return files after the last processed one
        new_files = all_files[last_idx + 1:]
        return [os.path.join(ingestions_dir, f) for f in new_files]
    except ValueError:
        # Last processed file not found, return all files
        return [os.path.join(ingestions_dir, f) for f in all_files]


def train_mood_model_incremental(dataset_path: str = "data/processed/mood_nexttrack_train.parquet",
                                model_path: str = "models/mood_classifier.pkl",
                                ingestions_dir: str = "data/ingestions",
                                new_trees: int = 5,
                                min_samples: int = 1,
                                test_size: float = 0.2,
                                random_state: int = 42) -> Dict:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Run full retrain first.")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    # Get last processed ingestion file from model metadata
    last_processed_file = None
    try:
        with open(model_path, 'rb') as f:
            existing_model_data = pickle.load(f)
            if "last_processed_ingestion_file" in existing_model_data:
                last_processed_file = existing_model_data["last_processed_ingestion_file"]
                print(f"Last processed ingestion file: {os.path.basename(last_processed_file)}")
    except Exception as e:
        print(f"Could not load last processed ingestion file: {e}")
    
    # Get new ingestion files (or history.parquet for first time)
    new_ingestion_files = get_new_ingestion_files(ingestions_dir, last_processed_file)
    
    if not new_ingestion_files:
        print("No new ingestion files found. Skipping incremental training.")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return {
            "train_accuracy": 0.0,
            "val_accuracy": 0.0,
            "train_f1_macro": 0.0,
            "val_f1_macro": 0.0,
            "train_roc_auc": 0.0,
            "val_roc_auc": 0.0,
            "n_features": len(model_data["feature_cols"]),
            "n_train": 0,
            "n_val": 0,
            "incremental": True,
            "skipped": True,
            "reason": "no_new_ingestion_files"
        }
    
    # Check if first time (using history.parquet)
    is_first_time = len(new_ingestion_files) == 1 and new_ingestion_files[0].endswith("history.parquet")
    
    if is_first_time:
        print("First incremental training: using all samples from history.parquet")
        df = pd.read_parquet(dataset_path)
        if df.empty:
            raise ValueError("Dataset is empty")
        new_df = df.copy()
    else:
        print(f"Found {len(new_ingestion_files)} new ingestion file(s)")
        
        # Load dataset and filter to samples from new ingestion files
        df = pd.read_parquet(dataset_path)
        
        if df.empty:
            raise ValueError("Dataset is empty")
        
        # Extract timestamps from ingestion filenames
        # Format: ingestion_YYYYMMDD_HHMMSS.parquet
        ingestion_timestamps = []
        for file_path in new_ingestion_files:
            filename = os.path.basename(file_path)
            # Extract timestamp from filename
            try:
                timestamp_str = filename.replace("ingestion_", "").replace(".parquet", "")
                # Parse YYYYMMDD_HHMMSS
                dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                ingestion_timestamps.append(dt)
            except ValueError:
                print(f"Warning: Could not parse timestamp from {filename}")
        
        if not ingestion_timestamps:
            print("Could not extract timestamps from ingestion files. Using dataset filtering fallback.")
            # Fallback to timestamp-based filtering
            if "last_track_timestamp" in df.columns:
                df["last_track_timestamp"] = pd.to_datetime(df["last_track_timestamp"])
                if last_processed_file:
                    # Try to extract timestamp from last processed file
                    try:
                        last_filename = os.path.basename(last_processed_file)
                        last_timestamp_str = last_filename.replace("ingestion_", "").replace(".parquet", "")
                        last_dt = datetime.strptime(last_timestamp_str, "%Y%m%d_%H%M%S")
                        new_df = df[df["last_track_timestamp"] > pd.to_datetime(last_dt)].copy()
                    except:
                        new_df = df.tail(50).copy()
                else:
                    new_df = df.tail(50).copy()
            else:
                new_df = df.tail(50).copy()
        else:
            # Filter dataset to samples from new ingestion files
            # Match by timestamp (samples created from tracks in these ingestion files)
            if "last_track_timestamp" in df.columns:
                df["last_track_timestamp"] = pd.to_datetime(df["last_track_timestamp"])
                min_timestamp = min(ingestion_timestamps)
                max_timestamp = max(ingestion_timestamps) + timedelta(hours=1)  # Add buffer
                new_df = df[
                    (df["last_track_timestamp"] >= pd.to_datetime(min_timestamp)) &
                    (df["last_track_timestamp"] <= pd.to_datetime(max_timestamp))
                ].copy()
                print(f"Found {len(new_df)} samples from new ingestion files")
            else:
                # Fallback: use last few rows
                new_df = df.tail(50).copy()
                print("No timestamp column found: using last samples as fallback")
    
    if len(new_df) < min_samples:
        print(f"Only {len(new_df)} new samples available (minimum: {min_samples}). Skipping incremental training.")
        # Return existing model metrics
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return {
            "train_accuracy": 0.0,
            "val_accuracy": 0.0,
            "train_f1_macro": 0.0,
            "val_f1_macro": 0.0,
            "train_roc_auc": 0.0,
            "val_roc_auc": 0.0,
            "n_features": len(model_data["feature_cols"]),
            "n_train": 0,
            "n_val": 0,
            "incremental": True,
            "skipped": True,
            "reason": "insufficient_new_samples"
        }
    
    print(f"Using {len(new_df)} new samples for incremental training")
    
    # Adjust number of trees based on sample size
    if len(new_df) < 5:
        new_trees = 1  # Very few samples, add just 1 tree
    elif len(new_df) < 10:
        new_trees = min(new_trees, 3)  # Few samples, add 2-3 trees
    else:
        new_trees = min(new_trees, 10)  # More samples, can add more trees
    
    print(f"Adding {new_trees} new trees to existing model")
    
    target_col = "target_mood_cluster"
    
    if target_col not in new_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    exclude_cols = [target_col, "date", "last_track_timestamp"] if "last_track_timestamp" in new_df.columns else [target_col, "date"]
    exclude_cols = [col for col in exclude_cols if col in new_df.columns]
    feature_cols = [col for col in new_df.columns if col not in exclude_cols]
    
    X = new_df[feature_cols].values
    y = new_df[target_col].values
    
    if pd.isna(X).any():
        print("Warning: Missing values in features, filling with 0")
        X = np.nan_to_num(X, nan=0.0)
    
    # For very small datasets, use all data for training (no validation split)
    if len(new_df) < 5:
        print(f"Very few samples ({len(new_df)}), using all for training (no validation split)")
        X_train, X_val = X, X
        y_train, y_val = y, y
    else:
        # Small split for validation
        from collections import Counter
        class_counts = Counter(y)
        min_class_count = min(class_counts.values())
        can_stratify = min_class_count >= 2 and len(new_df) >= 10
        
        if can_stratify:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=min(test_size, 0.3), random_state=random_state, stratify=y
            )
        else:
            # Use most for training, few for validation
            split_idx = max(1, int(len(X) * (1 - 0.3)))  # At least 1 sample for validation
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Load existing model to continue training
    existing_model = None
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                existing_model_data = pickle.load(f)
                existing_model = existing_model_data["model"]
                # Verify feature columns match
                if existing_model_data["feature_cols"] == feature_cols:
                    print(f"Found existing model with {existing_model.n_estimators} trees")
                    print("Continuing training with new data")
                else:
                    print("Feature columns changed, training from scratch")
                    existing_model = None
        except Exception as e:
            print(f"Could not load existing model: {e}, training from scratch")
    else:
        print("No existing model found, training from scratch")
    
    # Create new model with same hyperparameters
    # If we have an existing model, we'll continue training it
    model = xgb.XGBClassifier(
        n_estimators=new_trees,  # Add new trees to existing model
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state,
        eval_metric='mlogloss'
    )
    
    # Continue training from existing model if available
    if existing_model:
        print(f"Continuing training: adding {model.n_estimators} more trees to existing {existing_model.n_estimators} trees")
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            xgb_model=existing_model,  # Continue from existing model
            verbose=False
        )
        print(f"Model now has {model.n_estimators} total trees")
    else:
        print("Training new model from scratch")
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    y_train_proba = model.predict_proba(X_train)
    y_val_proba = model.predict_proba(X_val)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    train_f1 = f1_score(y_train, y_train_pred, average='macro')
    val_f1 = f1_score(y_val, y_val_pred, average='macro')
    
    # ROC-AUC for multi-class
    model_classes = model.classes_
    n_classes = len(model_classes)
    
    train_roc_auc = 0.0
    val_roc_auc = 0.0
    
    if n_classes >= 2:
        try:
            y_train_bin = label_binarize(y_train, classes=model_classes)
            y_val_bin = label_binarize(y_val, classes=model_classes)
            
            if n_classes == 2:
                train_roc_auc = roc_auc_score(y_train_bin[:, 1], y_train_proba[:, 1])
                val_roc_auc = roc_auc_score(y_val_bin[:, 1], y_val_proba[:, 1])
            else:
                train_roc_auc = roc_auc_score(y_train_bin, y_train_proba, multi_class='ovr', average='macro')
                val_roc_auc = roc_auc_score(y_val_bin, y_val_proba, multi_class='ovr', average='macro')
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not calculate ROC-AUC: {e}")
    
    metrics = {
        "train_accuracy": float(train_accuracy),
        "val_accuracy": float(val_accuracy),
        "train_f1_macro": float(train_f1),
        "val_f1_macro": float(val_f1),
        "train_roc_auc": float(train_roc_auc),
        "val_roc_auc": float(val_roc_auc),
        "n_features": len(feature_cols),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "incremental": True
    }
    
    print(f"Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")
    print(f"Train F1 (macro): {train_f1:.4f}, Val F1 (macro): {val_f1:.4f}")
    print(f"Train ROC-AUC: {train_roc_auc:.4f}, Val ROC-AUC: {val_roc_auc:.4f}")
    
    # Update last processed ingestion file
    # If first time (used history.parquet), find the most recent ingestion file to mark as processed
    if is_first_time:
        # Find most recent ingestion file to mark as processed
        if os.path.exists(ingestions_dir):
            all_files = [f for f in os.listdir(ingestions_dir) if f.startswith("ingestion_") and f.endswith(".parquet")]
            if all_files:
                all_files.sort()
                last_processed_file = os.path.join(ingestions_dir, all_files[-1])
            else:
                last_processed_file = None
        else:
            last_processed_file = None
    else:
        # Use the most recent ingestion file from new files
        last_processed_file = new_ingestion_files[-1] if new_ingestion_files else None
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        model_data = {
            "model": model,
            "feature_cols": feature_cols
        }
        if last_processed_file:
            model_data["last_processed_ingestion_file"] = last_processed_file
        pickle.dump(model_data, f)
    
    if last_processed_file:
        print(f"Updated last processed ingestion file: {os.path.basename(last_processed_file)}")
    print(f"Saved model to {model_path}")
    
    return metrics


if __name__ == "__main__":
    train_mood_model_incremental()

