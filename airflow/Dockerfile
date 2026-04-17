FROM apache/airflow:2.10.3-python3.10

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements-project.txt /tmp/requirements-project.txt
RUN pip install --no-cache-dir -r /tmp/requirements-project.txt
