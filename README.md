# Spotify ML Dashboard Backend

A headless ML backend that ingests Spotify listening data, trains prediction models, and exports dashboard data for visualization.

## Overview

This repository implements the complete ML pipeline for "My Spotify Brain":
- Data ingestion from Spotify Web API
- Feature engineering
- Model training (next-track mood prediction, hourly session-start prediction)
- Automated dashboard data export
- GitHub Actions integration for scheduled runs

## Project Structure

```
spotify-brain/
├── .github/workflows/      # GitHub Actions workflows
├── data/                   # Data storage
├── models/                 # Trained models (gitignored)
├── metrics/                # Training metrics
├── export/                 # Dashboard data export
└── src/                    # Source code
    ├── ingestion/          # Spotify API integration
    ├── features/           # Feature engineering
    ├── models/             # Model training
    ├── export/             # Dashboard export
    └── pipelines/          # Pipeline orchestration
```

## Setup

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   
   Create a `.env` file in the project root:
   ```bash
   cp .env.example .env
   ```
   
   Then edit `.env` and add your Spotify credentials:
   ```
   SPOTIFY_CLIENT_ID=your_client_id
   SPOTIFY_CLIENT_SECRET=your_client_secret
   SPOTIFY_REFRESH_TOKEN=your_refresh_token
   ```
   
   To get your refresh token, run:
   ```bash
   python get_refresh_token.py
   ```

3. **Run the pipeline:**
   ```bash
   python -m src.pipelines.run_full_retrain
   ```

### GitHub Actions Setup

Configure the following secrets in your repository settings:
- `SPOTIFY_CLIENT_ID`: Your Spotify app client ID
- `SPOTIFY_CLIENT_SECRET`: Your Spotify app client secret
- `SPOTIFY_REFRESH_TOKEN`: Your Spotify refresh token
- `WEBSITE_REPO_PAT`: Personal Access Token for the website repo

## Pipeline Overview

1. **Ingestion**: Fetch new tracks from Spotify API (only newer than latest in dataset)
2. **Feature Engineering**: Build mood clusters, datasets, and genre profiles
3. **Model Training**: Train mood and session-start classifiers
4. **Export**: Generate `dashboard_data.json` for the website

## Output

The pipeline exports `export/dashboard_data.json` containing:
- Next-track mood predictions
- Hourly session-start probabilities
- Mood cluster definitions and centroids
- Genre profiles
- Performance metrics

This file is automatically synced to the personal website repository.
