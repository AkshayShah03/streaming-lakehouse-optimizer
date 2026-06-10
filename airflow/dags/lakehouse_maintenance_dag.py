"""Airflow DAG: scheduled, guarded layout maintenance.

Runs the closed loop every 30 minutes:
  collect workload features -> model proposes -> shadow-eval -> guard ->
  apply iff promoted -> log to ledger.

In production the feature-collection task reads Trino's query log + Iceberg
metadata; the apply task uses PyIcebergBackend against the REST catalog.
The guard makes this safe to run unattended -- it can always decline.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


default_args = {
    "owner": "data-platform",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def _collect_features(**ctx):
    """Read query log + Iceberg metadata -> WorkloadFeatures (pushed to XCom)."""
    # Production: query Trino system.runtime.queries + Iceberg `files` table.
    # Returned as a plain dict so it serializes through XCom.
    from lakehouse.metrics.table_stats import TableStats  # noqa
    raise NotImplementedError("wire to Trino query log + Iceberg metadata reader")


def _optimize_and_guard(**ctx):
    from lakehouse.config import settings
    from lakehouse.maintenance.iceberg_ops import PyIcebergBackend
    from lakehouse.maintenance.orchestrator import run_once
    from lakehouse.optimizer.cost_model import CostModel
    # model = CostModel.load(settings.model_path)
    # backend = PyIcebergBackend(settings.iceberg_rest_uri, settings.table_id,
    #                            settings.warehouse)
    # production = <hydrate TableStats from catalog>
    # result = run_once(model=model, production=production, backend=backend,
    #                   wf=<from XCom>, ledger_path=settings.ledger_path)
    raise NotImplementedError("hydrate production table + run_once")


with DAG(
    dag_id="lakehouse_layout_maintenance",
    description="Workload-learned, guarded compaction & layout optimization",
    start_date=datetime(2026, 1, 1),
    schedule="*/30 * * * *",
    catchup=False,
    default_args=default_args,
    tags=["lakehouse", "iceberg", "optimizer"],
) as dag:
    collect = PythonOperator(task_id="collect_workload_features",
                             python_callable=_collect_features)
    optimize = PythonOperator(task_id="optimize_and_guard",
                              python_callable=_optimize_and_guard)
    collect >> optimize
