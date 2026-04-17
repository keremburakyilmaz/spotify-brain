# Airflow (local / dev)

Local Airflow that mirrors the two GitHub Actions workflows, with each pipeline
step exposed as its own task so you get per-step retries, logs, and backfills.

- `spotify_update` - runs every 30 min (`0,30 * * * *`), mirrors `update.yml`.
- `spotify_full_retrain` - runs daily at 00:15 UTC (`15 0 * * *`), mirrors `full-retrain.yml`.

GitHub Actions remains the production scheduler. The website-repo dashboard
push is intentionally **not** in the DAGs - it stays in GH Actions until this
setup has been validated in practice.

## Quickstart

From `airflow/`:

```bash
cp .env.example .env
# fill SPOTIFY_CLIENT_ID / _SECRET / _REFRESH_TOKEN in .env

docker compose up -d --build
```

Open <http://localhost:8080> (admin / admin). Unpause the two DAGs.

The repo root is bind-mounted at `/opt/spotify-brain` inside the containers, so
the DAGs read and write the same `data/`, `models/`, `export/`, and `metrics/`
files your local checkout uses.

## DAG shape

Update:

```
ingest → assign_clusters → predict_mood → update_history ┬→ build_dashboard
                                                         └→ detect_drift → log_metrics
```

Full retrain:

```
verify_history → build_mood_clusters ┬→ build_mood_dataset    → train_mood_model    ┐
                                     ├→ build_session_dataset → train_session_model ├→ log_metrics → build_dashboard
                                     └→ detect_drift ───────────────────────────────┘
```

If ingest returns no new tracks, it raises `AirflowSkipException` and the rest
of the update DAG skips.

## Notes

- Executor: `LocalExecutor` on Postgres. Enough to run the parallel dataset/train
  branches of the retrain DAG; no Celery/Redis overhead.
- Both DAGs import the existing modules under `src/` rather than re-implementing
  the steps - see `dags/common.py` for the `sys.path` + `chdir` shim.
- `logs/` and `.env` are gitignored.
