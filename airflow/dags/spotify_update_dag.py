from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.operators.python import PythonOperator

from common import enter_project

HISTORY_PATH = "data/history.parquet"


def _ingest(**_):
    enter_project()
    from ingestion.spotify_ingest import ingest
    from features.build_mood_clusters_incremental import get_latest_ingestion_file

    df = ingest()
    if df.empty:
        raise AirflowSkipException("No new tracks to process")

    path = get_latest_ingestion_file()
    if not path:
        raise AirflowSkipException("No ingestion file produced")
    return path


def _assign_clusters(**context):
    enter_project()
    from features.build_mood_clusters_incremental import assign_clusters_to_ingestion_file

    path = context["ti"].xcom_pull(task_ids="ingest")
    assign_clusters_to_ingestion_file(path)
    return path


def _predict_mood(**context):
    enter_project()
    from pipelines.run_update import predict_mood_for_tracks

    path = context["ti"].xcom_pull(task_ids="assign_clusters")
    try:
        predict_mood_for_tracks(path)
    except Exception as e:
        # Matches the tolerance in run_update.py - mood predict is best-effort.
        print(f"Mood prediction skipped: {e}")
    return path


def _update_history(**context):
    enter_project()
    import pandas as pd

    from ingestion.spotify_ingest import update_history_from_ingestion

    path = context["ti"].xcom_pull(task_ids="predict_mood")
    update_history_from_ingestion(path)

    if not os.path.exists(HISTORY_PATH):
        return {"num_tracks": 0, "num_sessions": 0}
    history = pd.read_parquet(HISTORY_PATH)
    return {
        "num_tracks": int(len(history)),
        "num_sessions": int(history["session_id"].nunique()) if "session_id" in history.columns else 0,
    }


def _build_dashboard(**_):
    enter_project()
    from export.build_dashboard_json import build_dashboard_json

    build_dashboard_json()


def _detect_drift(**_):
    enter_project()
    from models.detect_drift import detect_drift

    return detect_drift(HISTORY_PATH)


def _log_metrics(**context):
    enter_project()
    from models.log_metrics import log_metrics

    counts = context["ti"].xcom_pull(task_ids="update_history") or {}
    drift = context["ti"].xcom_pull(task_ids="detect_drift")
    log_metrics(
        num_tracks=counts.get("num_tracks", 0),
        num_sessions=counts.get("num_sessions", 0),
        mood_model_metrics=None,
        session_model_metrics=None,
        drift_data=drift,
    )


default_args = {
    "owner": "spotify-brain",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="spotify_update",
    description="Incremental Spotify ingest, cluster assignment, dashboard refresh",
    start_date=datetime(2026, 1, 1),
    schedule="0,30 * * * *",
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["spotify-brain"],
) as dag:
    ingest = PythonOperator(task_id="ingest", python_callable=_ingest)
    assign_clusters = PythonOperator(task_id="assign_clusters", python_callable=_assign_clusters)
    predict_mood = PythonOperator(task_id="predict_mood", python_callable=_predict_mood)
    update_history = PythonOperator(task_id="update_history", python_callable=_update_history)
    build_dashboard = PythonOperator(task_id="build_dashboard", python_callable=_build_dashboard)
    detect_drift = PythonOperator(task_id="detect_drift", python_callable=_detect_drift)
    log_metrics_task = PythonOperator(task_id="log_metrics", python_callable=_log_metrics)

    ingest >> assign_clusters >> predict_mood >> update_history
    update_history >> [build_dashboard, detect_drift]
    detect_drift >> log_metrics_task
