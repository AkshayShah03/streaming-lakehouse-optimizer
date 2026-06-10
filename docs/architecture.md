# Architecture

## Data flow

```
                        exactly-once (checkpoint-aligned 2PC)
 Postgres ──Debezium──▶ Kafka ──▶ Flink ─────────────────────▶ Iceberg (S3/MinIO)
 (source)   (CDC/WAL)  (topic)  (PyFlink)                       │  ▲
                                                                │  │ rewrite_data_files
                                              Trino / Spark ◀────┘  │ (only if guard promotes)
                                              (query engines)       │
                                                                    │
   ┌────────────────────────  MAINTENANCE CONTROL LOOP  ───────────┘
   │
   │  1. collect workload features  (Trino query log + Iceberg `files` metadata)
   │  2. cost model proposes ranked layouts            [optimizer/]
   │  3. shadow-evaluate top-k on a clone (measure)    [maintenance/shadow_eval]
   │  4. regression guard decides (zero-regression)    [maintenance/regression_guard]
   │  5. apply iff promoted; append decision to ledger [maintenance/iceberg_ops]
   └─ scheduled by Airflow every 30m                   [airflow/dags/]
```

## Why each guardrail exists

| Failure mode | Guardrail | Where |
|---|---|---|
| Streaming creates millions of tiny files | learned compaction targeting | `optimizer/`, `maintenance/` |
| Model proposes a bad layout | shadow-eval measures before promote | `maintenance/shadow_eval.py` |
| A change is cheaper but slower | per-metric regression tolerances | `maintenance/regression_guard.py` |
| Over-aggressive compaction churns bytes | write-amplification as a guarded metric | `metrics/table_stats.py` |
| Compaction loses/dupes rows | row-count invariant in shadow + apply | `shadow_eval`, `iceberg_ops` |
| Duplicate rows on Flink recovery | checkpoint-aligned idempotent commit | `ingest/exactly_once.py` |
| Producer breaks the schema | backward-compat evolution gate (CI) | `catalog/schema_evolution.py` |
| Silent objective drift | one benchmark suite for train + eval | `maintenance/benchmark.py` |

## Simulation vs. production

Everything in `tests/` and `scripts/run_local_demo.py` runs against
`SimulatedIcebergTable`, a deterministic *physical* model of small-file
accumulation and scan cost. It exists so the optimizer + guard loop is fully
testable in CI with no cluster. The same interfaces (`apply_layout`, `scan`,
`stats`) are implemented by `PyIcebergBackend` against a real REST catalog;
switching backends is the only change between local and production.
