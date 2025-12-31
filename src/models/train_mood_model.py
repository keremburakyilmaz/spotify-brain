import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import xgboost as xgb
from typing import Dict, Tuple


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
    
    exclude_cols = [target_col, "date"] if "date" in df.columns else [target_col]
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
    
    if use_time_split and "date" in df.columns:
        df_sorted = df.sort_values("date").reset_index(drop=True)
        split_idx = int(len(df_sorted) * (1 - test_size))
        
        X_train = df_sorted.loc[:split_idx, feature_cols].values
        y_train = df_sorted.loc[:split_idx, target_col].values
        X_val = df_sorted.loc[split_idx:, feature_cols].values
        y_val = df_sorted.loc[split_idx:, target_col].values
        
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0)
    else:
        # Check if stratification is possible (each class needs at least 2 samples)
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
    
    train_classes = np.unique(y_train)
    val_classes = np.unique(y_val)
    missing_in_train = set(unique_classes) - set(train_classes)
    
    if len(missing_in_train) > 0:
        print(f"Ensuring all {n_classes} classes are in training set (moving samples from validation)...")
        print(f"  Training classes before: {sorted(train_classes)}")
        print(f"  Missing classes: {sorted(missing_in_train)}")
        # Convert to DataFrame for easier manipulation
        train_df = pd.DataFrame(X_train, columns=feature_cols)
        train_df['target'] = y_train
        val_df = pd.DataFrame(X_val, columns=feature_cols)
        val_df['target'] = y_val
        
        # For each missing class, move one sample from validation to training
        for missing_class in missing_in_train:
            # Find samples of this class in validation
            class_samples = val_df[val_df['target'] == missing_class]
            if len(class_samples) > 0:
                # Move first sample to training
                sample_to_move = class_samples.iloc[0:1]
                train_df = pd.concat([train_df, sample_to_move], ignore_index=True)
                val_df = val_df.drop(sample_to_move.index)
                print(f"  Moved 1 sample of class {missing_class} from validation to training")
        
        # Convert back to numpy arrays
        X_train = train_df[feature_cols].values
        y_train = train_df['target'].values
        X_val = val_df[feature_cols].values
        y_val = val_df['target'].values
        
        print(f"  Training classes after: {sorted(np.unique(y_train))}")
        print(f"After adjustment - Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
    print(f"Train classes: {sorted(np.unique(y_train))}, Val classes: {sorted(np.unique(y_val))}")
    
    print("Training XGBoost classifier")
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state,
        eval_metric='mlogloss',
        objective='multi:softprob',  # Explicitly set multi-class objective
        num_class=n_classes  # Explicitly set number of classes
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
        "n_features": len(feature_cols),
        "n_train": len(X_train),
        "n_val": len(X_val)
    }
    
    print(f"Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")
    print(f"Train F1 (macro): {train_f1:.4f}, Val F1 (macro): {val_f1:.4f}")
    print(f"Train ROC-AUC: {train_roc_auc:.4f}, Val ROC-AUC: {val_roc_auc:.4f}")
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump({
            "model": model,
            "feature_cols": feature_cols
        }, f)
    
    print(f"Saved model to {model_path}")
    
    return metrics

if __name__ == "__main__":
    train_mood_model()






