"""Central config (12-factor: env-overridable)."""
from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    kafka_bootstrap: str = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
    cdc_topic: str = os.getenv("CDC_TOPIC", "fleet.public.telemetry")
    iceberg_rest_uri: str = os.getenv("ICEBERG_REST_URI", "http://iceberg-rest:8181")
    warehouse: str = os.getenv("ICEBERG_WAREHOUSE", "s3://warehouse/")
    table_id: str = os.getenv("ICEBERG_TABLE", "fleet.telemetry")
    model_path: str = os.getenv("COST_MODEL_PATH", "artifacts/cost_model.joblib")
    ledger_path: str = os.getenv("LEDGER_PATH", "artifacts/promotion_ledger.jsonl")


settings = Settings()
