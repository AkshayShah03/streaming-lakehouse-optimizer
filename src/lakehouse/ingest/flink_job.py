"""PyFlink job: Kafka (Debezium CDC) -> Iceberg, exactly-once.

This is the production ingest path. It is not executed in unit tests (it
needs a Flink cluster + Iceberg catalog, provided by docker-compose), but it
is the real job submitted by `make flink-submit`. The exactly-once guarantee
comes from:
  * checkpointing enabled with EXACTLY_ONCE mode
  * Kafka source committing offsets on checkpoint
  * Iceberg sink performing checkpoint-aligned two-phase commits

The dedup/commit invariant this relies on is unit-tested in
`lakehouse.ingest.exactly_once`.
"""
from __future__ import annotations


def build_job(checkpoint_interval_ms: int = 30_000) -> None:
    from pyflink.datastream import StreamExecutionEnvironment, CheckpointingMode
    from pyflink.table import StreamTableEnvironment, EnvironmentSettings

    env = StreamExecutionEnvironment.get_execution_environment()
    env.enable_checkpointing(checkpoint_interval_ms, CheckpointingMode.EXACTLY_ONCE)
    env.get_checkpoint_config().set_min_pause_between_checkpoints(10_000)
    env.get_checkpoint_config().set_checkpoint_timeout(120_000)
    env.get_checkpoint_config().enable_unaligned_checkpoints()

    t_env = StreamTableEnvironment.create(
        env, environment_settings=EnvironmentSettings.in_streaming_mode())

    # 1) Source: Debezium-formatted CDC topic.
    t_env.execute_sql("""
        CREATE TABLE telemetry_cdc (
            vehicle_id   BIGINT,
            ts_ms        BIGINT,
            speed_kph    DOUBLE,
            battery_pct  DOUBLE,
            motor_temp_c DOUBLE,
            pack_voltage DOUBLE,
            odometer_km  DOUBLE,
            lat          DOUBLE,
            lon          DOUBLE,
            event_time   AS TO_TIMESTAMP_LTZ(ts_ms, 3),
            WATERMARK FOR event_time AS event_time - INTERVAL '5' SECOND
        ) WITH (
            'connector' = 'kafka',
            'topic' = 'fleet.public.telemetry',
            'properties.bootstrap.servers' = 'kafka:9092',
            'properties.group.id' = 'flink-iceberg-sink',
            'scan.startup.mode' = 'earliest-offset',
            'format' = 'debezium-json'
        )
    """)

    # 2) Iceberg catalog (REST) + target table.
    t_env.execute_sql("""
        CREATE CATALOG iceberg WITH (
            'type' = 'iceberg',
            'catalog-type' = 'rest',
            'uri' = 'http://iceberg-rest:8181',
            'warehouse' = 's3://warehouse/',
            's3.endpoint' = 'http://minio:9000'
        )
    """)
    t_env.execute_sql("CREATE DATABASE IF NOT EXISTS iceberg.fleet")
    t_env.execute_sql("""
        CREATE TABLE IF NOT EXISTS iceberg.fleet.telemetry (
            vehicle_id   BIGINT,
            ts_ms        BIGINT,
            speed_kph    DOUBLE,
            battery_pct  DOUBLE,
            motor_temp_c DOUBLE,
            pack_voltage DOUBLE,
            odometer_km  DOUBLE,
            lat          DOUBLE,
            lon          DOUBLE,
            event_time   TIMESTAMP_LTZ(3)
        ) PARTITIONED BY (hours(event_time)) WITH (
            'format-version' = '2',
            'write.target-file-size-bytes' = '134217728',
            'write.distribution-mode' = 'hash'
        )
    """)

    # 3) Stream insert. Iceberg sink commits per checkpoint (exactly-once).
    t_env.execute_sql("""
        INSERT INTO iceberg.fleet.telemetry
        SELECT vehicle_id, ts_ms, speed_kph, battery_pct, motor_temp_c,
               pack_voltage, odometer_km, lat, lon, event_time
        FROM telemetry_cdc
    """)


if __name__ == "__main__":
    build_job()
