from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.exceptions import AirflowFailException
from airflow.operators.python import PythonOperator

from common import enter_project

HISTORY_PATH = "data/history.parquet"


def _verify_history(**_):
    enter_project()
    import pandas as pd

    if not os.path.exists(HISTORY_PATH):
        raise AirflowFailException(f"{HISTORY_PATH} missing - run the update DAG first")
    df = pd.read_parquet(HISTORY_PATH)
    if df.empty:
        raise AirflowFailException(f"{HISTORY_PATH} is empty")
    return {
        "num_tracks": int(len(df)),
        "num_sessions": int(df["session_id"].nunique()) if "session_id" in df.columns else 0,
    }


def _build_mood_clusters(**_):
    enter_project()
    from features.build_mood_clusters import build_mood_clusters

    build_mood_clusters()


def _build_mood_dataset(**_):
    enter_project()
    from features.build_mood_dataset import build_mood_dataset

    build_mood_dataset()


def _build_session_dataset(**_):
    enter_project()
    from features.build_session_dataset import build_session_dataset

    build_session_dataset()


def _train_mood_model(**_):
    enter_project()
    from models.train_mood_model import train_mood_model

    return train_mood_model()


def _train_session_model(**_):
    enter_project()
    from models.train_session_model import train_session_model

    return train_session_model(rolling_window_days=90)


def _detect_drift(**_):
    enter_project()
    from models.detect_drift import detect_drift

    return detect_drift(HISTORY_PATH)


def _log_metrics(**context):
    enter_project()
    from models.log_metrics import log_metrics

    counts = context["ti"].xcom_pull(task_ids="verify_history") or {}
    log_metrics(
        num_tracks=counts.get("num_tracks", 0),
        num_sessions=counts.get("num_sessions", 0),
        mood_model_metrics=context["ti"].xcom_pull(task_ids="train_mood_model"),
        session_model_metrics=context["ti"].xcom_pull(task_ids="train_session_model"),
        drift_data=context["ti"].xcom_pull(task_ids="detect_drift"),
    )


def _build_dashboard(**_):
    enter_project()
    from export.build_dashboard_json import build_dashboard_json

    build_dashboard_json()


default_args = {
    "owner": "spotify-brain",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="spotify_full_retrain",
    description="Rebuild mood clusters, datasets, and retrain both classifiers",
    start_date=datetime(2026, 1, 1),
    schedule="15 0 * * *",
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["spotify-brain"],
) as dag:
    verify_history = PythonOperator(task_id="verify_history", python_callable=_verify_history)
    build_clusters = PythonOperator(task_id="build_mood_clusters", python_callable=_build_mood_clusters)
    build_mood_ds = PythonOperator(task_id="build_mood_dataset", python_callable=_build_mood_dataset)
    build_session_ds = PythonOperator(task_id="build_session_dataset", python_callable=_build_session_dataset)
    train_mood = PythonOperator(task_id="train_mood_model", python_callable=_train_mood_model)
    train_session = PythonOperator(task_id="train_session_model", python_callable=_train_session_model)
    detect_drift = PythonOperator(task_id="detect_drift", python_callable=_detect_drift)
    log_metrics_task = PythonOperator(task_id="log_metrics", python_callable=_log_metrics)
    build_dashboard = PythonOperator(task_id="build_dashboard", python_callable=_build_dashboard)

    verify_history >> build_clusters
    build_clusters >> [build_mood_ds, build_session_ds, detect_drift]
    build_mood_ds >> train_mood
    build_session_ds >> train_session
    [train_mood, train_session, detect_drift] >> log_metrics_task >> build_dashboard
