# Spotify Brain

A machine learning system that analyzes Spotify listening history to predict mood clusters and session start times. The system uses audio features from tracks to build mood-based clusters and trains models to predict listening patterns.

## Table of Contents

- [Overview](#overview)
- [Data Ingestion](#data-ingestion)
- [Features](#features)
- [Data Storage](#data-storage)
- [Mood Clustering](#mood-clustering)
- [Dataset Creation](#dataset-creation)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Workflows](#workflows)

## Overview

The Spotify Brain system consists of two main pipelines:

1. **Update Pipeline**: Incrementally ingests new tracks, assigns them to existing mood clusters, and updates the dashboard
2. **Full Retrain Pipeline**: Rebuilds mood clusters from scratch, creates training datasets, and retrains both models

## Data Ingestion

### How We Are Ingesting

The ingestion process (`src/ingestion/spotify_ingest.py`) fetches recently played tracks from the Spotify API and enriches them with audio features and metadata:

1. **Spotify API Authentication**: Uses OAuth2 refresh tokens to authenticate with Spotify's API
2. **Fetch Recently Played**: Retrieves up to 50 recently played tracks using the `/v1/me/player/recently-played` endpoint
3. **Incremental Updates**: Only fetches tracks played after the latest timestamp in `history.parquet` to avoid duplicates
4. **Audio Features**: Fetches audio features from the ReccoBeats API (`https://api.reccobeats.com/v1/audio-features`)
5. **Track Metadata**: Fetches track names, artist names, and album artwork from Spotify's `/v1/tracks` endpoint
6. **Session Assignment**: Groups tracks into listening sessions based on a 30-minute gap threshold

### Which Features We Are Ingesting

The system ingests the following features for each track:

**Basic Track Information:**
- `track_id`: Spotify track ID
- `track_name`: Name of the track
- `artist_name`: Comma-separated list of artist names
- `image_url`: URL to album artwork (prefers 300x300, falls back to largest available)
- `played_at`: Timestamp when the track was played

**Audio Features** (from ReccoBeats API):
- `valence`: Musical positiveness (0.0 = sad/depressing, 1.0 = happy/euphoric)
- `energy`: Perceptual intensity and power (0.0 = low energy, 1.0 = high energy)
- `danceability`: Suitability for dancing (0.0 = not danceable, 1.0 = very danceable)
- `acousticness`: Confidence that the track is acoustic (0.0 = not acoustic, 1.0 = high confidence acoustic)
- `instrumentalness`: Predicts whether a track contains no vocals (0.0 = vocal content, 1.0 = instrumental)
- `tempo`: Overall estimated tempo in beats per minute (BPM)
- `tempo_norm`: Normalized tempo (tempo / 250.0, capped at 1.0)

**Derived Features:**
- `session_id`: Assigned based on 30-minute gaps between tracks

## Data Storage

### Ingestion Files Logic

Each ingestion run creates a timestamped file in `data/ingestions/`:

- **Format**: `ingestion_YYYYMMDD_HHMMSS.parquet`
- **Purpose**: Preserves a snapshot of newly ingested tracks before they're merged into history
- **Contents**: All newly ingested tracks with their features, metadata, and session IDs
- **Lifecycle**:
  1. Created during ingestion with basic track info and audio features
  2. Updated with `session_id` after session assignment
  3. Updated with `mood_cluster_id` during the update pipeline
  4. Used to update `history.parquet` at the end of the update pipeline

This approach allows:
- **Audit Trail**: Track exactly when data was ingested
- **Recovery**: Recover from failed updates by re-processing ingestion files
- **Incremental Processing**: Process new tracks separately before merging

### history.parquet Logic

The main data file (`data/history.parquet`) contains the complete listening history:

- **Purpose**: Single source of truth for all historical listening data
- **Update Strategy**: Incrementally updated by appending new tracks from ingestion files
- **Data Quality**:
  - All rows must have complete features (no null values)
  - Duplicates are removed based on `(track_id, played_at)` pairs
  - Tracks are sorted by `played_at` timestamp
- **Schema**: Contains all ingested features plus:
  - `mood_cluster_id`: Assigned cluster ID
  - `session_id`: Listening session identifier

**Update Process**:
1. Load existing `history.parquet`
2. Load latest ingestion file
3. Merge and deduplicate
4. Remove rows with null values
5. Save updated history

## Mood Clustering

### How We Are Creating the Mood Clusters

Mood clusters are created using K-Means clustering on audio features (`src/features/build_mood_clusters.py`):

1. **Feature Extraction**: Extract 6 audio features from history:
   - `valence`, `energy`, `danceability`, `acousticness`, `instrumentalness`, `tempo_norm`

2. **Optimal K Selection**: Automatically determine the optimal number of clusters (K) using:
   - **Silhouette Score**: Primary method - measures how similar objects are to their own cluster vs. other clusters
   - **Elbow Method**: Fallback - finds the point of maximum curvature in the inertia plot
   - **Range**: Tests K values from 3 to 15 (or up to `n_samples // 10`)

3. **K-Means Clustering**: Fit K-Means with the optimal K value on all tracks with complete features

4. **Cluster Assignment**: Assign each track to its nearest cluster centroid

5. **Cluster Metadata**: Generate human-readable labels for each cluster based on centroid values:
   - Energy level: High/Medium/Low
   - Valence: positive/neutral/negative
   - Acousticness: acoustic/electronic

6. **Output**: Save cluster metadata to `models/mood_clusters.json` with:
   - Cluster IDs
   - Centroid coordinates
   - Human-readable labels

### Incremental Cluster Assignment

For new tracks in the update pipeline (`src/features/build_mood_clusters_incremental.py`):

1. Load existing cluster centroids from `models/mood_clusters.json`
2. For each new track with complete features:
   - Calculate Euclidean distance to all cluster centroids
   - Assign to the nearest cluster
3. Update the ingestion file with cluster assignments

This allows new tracks to be assigned to existing clusters without rebuilding all clusters.

## Dataset Creation

### Mood Dataset (`mood_nexttrack_train.parquet`)

The mood prediction dataset (`src/features/build_mood_dataset.py`) predicts the mood cluster of the next track in a session:

**Window-Based Approach**:
- Uses a sliding window of the last N tracks (default: 3) to predict the next track's mood
- Creates one training sample per position in each session

**Features Extracted**:

1. **Sequence Features**:
   - `mood_cluster_0`, `mood_cluster_1`, `mood_cluster_2`: Mood cluster IDs of the last 3 tracks

2. **Aggregated Audio Features** (from last N tracks):
   - `{feature}_mean`: Mean of each audio feature over the window
   - `{feature}_std`: Standard deviation of each audio feature over the window
   - Features: `valence`, `energy`, `danceability`, `acousticness`, `instrumentalness`, `tempo_norm`

3. **Time Features** (cyclical encoding):
   - `hour_sin`, `hour_cos`: Hour of day (0-23) encoded as sin/cos
   - `day_sin`, `day_cos`: Day of week (0-6) encoded as sin/cos
   - `is_weekend`: Binary indicator (1 if Saturday or Sunday)

4. **Session Context**:
   - `session_position`: Position of the current track in the session
   - `session_length`: Total number of tracks in the session
   - `time_since_session_start`: Minutes since the session started

5. **Current Track Features**:
   - `current_{feature}`: Audio features of the most recent track in the window

**Target**: `target_mood_cluster` - The mood cluster ID of the next track

### Session Dataset (`session_start_train.parquet`)

The session start prediction dataset (`src/features/build_session_dataset.py`) predicts whether a listening session will start in a given hour:

**Time-Based Approach**:
- Creates one sample per hour in the date range covered by history
- Predicts binary outcome: session started (1) or not (0) in that hour

**Features Extracted**:

1. **Time Features**:
   - `date`: Date of the hour
   - `hour`: Hour of day (0-23)
   - `hour_sin`, `hour_cos`: Cyclical encoding of hour
   - `day_sin`, `day_cos`: Cyclical encoding of day of week
   - `day_of_month`: Day of month (1-31)
   - `is_weekend`: Binary indicator

2. **Historical Patterns**:
   - `rolling_listening_frequency_7d`: Number of sessions in this hour over the last 7 days
   - `rolling_listening_frequency_30d`: Number of sessions in this hour over the last 30 days
   - `time_since_last_session`: Hours since the last session started
   - `avg_session_duration_last_7d`: Average session duration (minutes) over last 7 days
   - `total_tracks_last_7d`: Total tracks played in last 7 days
   - `total_tracks_last_30d`: Total tracks played in last 30 days

**Target**: `target_session_start` - Binary indicator (1 if a session started in this hour, 0 otherwise)

## Model Training

### How We Are Training the Models

Both models use XGBoost classifiers with similar training procedures:

#### Mood Model Training (`src/models/train_mood_model.py`)

1. **Data Loading**: Load `mood_nexttrack_train.parquet`
2. **Data Splitting**:
   - **Time-based split** (default): Split by date to avoid temporal leakage
   - **Random split** (fallback): Stratified random split if time-based not possible
   - Train/validation split: 80/20
3. **Class Balancing**: Ensures all classes are present in training set (moves samples from validation if needed)
4. **Model Configuration**:
   - Algorithm: XGBoost Classifier
   - Objective: `multi:softprob` (multi-class classification)
   - Number of estimators: 100
   - Max depth: 6
   - Learning rate: 0.1
   - Random state: 42
5. **Training**: Fit model with early stopping on validation set
6. **Evaluation**: Calculate accuracy, F1-score (macro), and ROC-AUC (one-vs-rest)
7. **Saving**: Save model and feature column names to `models/mood_classifier.pkl`

#### Session Model Training (`src/models/train_session_model.py`)

1. **Data Loading**: Load `session_start_train.parquet`
2. **Rolling Window** (optional): Filter to last N days (default: 90 days) to focus on recent patterns
3. **Data Splitting**:
   - **Time-based split** (default): Split by date
   - **Random split** (fallback): Stratified random split
   - Train/validation split: 80/20
4. **Model Configuration**:
   - Algorithm: XGBoost Classifier
   - Objective: Binary classification (logistic)
   - Number of estimators: 100
   - Max depth: 6
   - Learning rate: 0.1
   - Random state: 42
5. **Training**: Fit model with early stopping on validation set
6. **Evaluation**: Calculate accuracy and ROC-AUC, plus comprehensive evaluation with baselines
7. **Saving**: Save model, feature columns, and metadata to `models/session_classifier.pkl`

## Evaluation

### What Evaluations We Are Doing and How

#### Mood Model Evaluation

**Metrics Calculated**:
- **Accuracy**: Percentage of correct predictions
- **F1-Score (Macro)**: Average F1-score across all classes
- **ROC-AUC**: One-vs-rest ROC-AUC with macro averaging

**Evaluation Process**:
1. Predict on training and validation sets
2. Calculate metrics for both sets
3. Log metrics to `metrics/metrics_history.json`

#### Session Model Evaluation (`src/models/evaluate_session_model.py`)

**Comprehensive Evaluation** includes:

1. **Model Metrics**:
   - **PR-AUC** (Primary): Precision-Recall AUC (better for imbalanced data)
   - **ROC-AUC**: Receiver Operating Characteristic AUC
   - **F1-Score**: Harmonic mean of precision and recall
   - **F1 at Multiple Thresholds**: F1-score at thresholds 0.1-0.9
   - **Calibration Error**: Mean absolute error between predicted and actual probabilities

2. **Baseline Comparisons**:
   - **Always Negative Baseline**: Predicts no session starts (all zeros)
   - **Historical Hour Prior**: Predicts based on historical frequency of sessions in each hour

3. **Per-Hour Metrics**: Performance broken down by time of day:
   - Morning (6-11)
   - Afternoon (12-17)
   - Evening (18-23)
   - Night (0-5)

4. **Comparison Summary**:
   - Whether model beats each baseline
   - Relative performance metrics

**Evaluation Process**:
1. Calculate model metrics on validation set
2. Generate baseline predictions
3. Calculate baseline metrics
4. Compare model vs. baselines
5. Calculate per-hour performance
6. Log all results to metrics history

### Drift Detection (`src/models/detect_drift.py`)

The system monitors for data drift by comparing recent data (last 7 days) to prior data (30 days before that):

**Feature Drift Metrics**:
- **Session Duration**: Mean and std of session durations
- **Sessions Per Day**: Mean and std of daily session counts
- **Tracks Per Day**: Mean and std of daily track counts
- **Start Hour Entropy**: Shannon entropy of session start hour distribution

**Label Drift Metrics**:
- **Session Start Rate**: Average sessions per day
- **Days With Sessions**: Fraction of days with at least one session
- **Per-Hour Frequency**: Session start frequency for each hour (0-23)

**Drift Detection**:
- Flags significant drift if any metric changes by >20%
- Logs percentage changes for all metrics
- Records drift detection timestamp

## Workflows

### Update Pipeline (`.github/workflows/update.yml`)

**Schedule**: Runs every 30 minutes (at :00 and :30 UTC)

**Steps**:
1. **Ingest New Tracks**: Fetch recently played tracks from Spotify
2. **Assign Clusters**: Assign new tracks to existing mood clusters using nearest centroid
3. **Predict Moods**: Use mood model to predict cluster for tracks with sufficient context
4. **Update History**: Merge new tracks into `history.parquet` (removing nulls)
5. **Build Dashboard**: Generate dashboard JSON for visualization
6. **Detect Drift**: Monitor for data drift and log metrics
7. **Deploy Dashboard**: Copy dashboard data to personal website repository
8. **Commit Changes**: Push all updates to repository

**Key Characteristics**:
- Incremental: Only processes new tracks
- Fast: No model retraining
- Non-destructive: Preserves ingestion files for audit

### Full Retrain Pipeline (`.github/workflows/full-retrain.yml`)

**Schedule**: Runs daily at 12:15 AM UTC

**Steps**:
1. **Load History**: Read existing `history.parquet`
2. **Build Mood Clusters**: Rebuild clusters from scratch using all historical data
3. **Build Mood Dataset**: Create training dataset for mood prediction
4. **Build Session Dataset**: Create training dataset for session prediction
5. **Train Mood Model**: Retrain mood classifier on updated dataset
6. **Train Session Model**: Retrain session classifier with 90-day rolling window
7. **Detect Drift**: Monitor for data drift and log metrics
8. **Build Dashboard**: Generate dashboard JSON
9. **Deploy Dashboard**: Copy dashboard data to personal website repository
10. **Commit Changes**: Push all updates including new models

**Key Characteristics**:
- Complete: Rebuilds everything from scratch
- Adaptive: Clusters adapt to changing listening patterns
- Comprehensive: Updates all models and datasets

### Workflow Configuration

Both workflows:
- Use GitHub Actions with Ubuntu runners
- Set up Conda environment from `environment.yml`
- Install Python dependencies from `requirements.txt`
- Use secrets for Spotify API credentials
- Deploy dashboard data to a separate website repository
- Commit and push all changes automatically