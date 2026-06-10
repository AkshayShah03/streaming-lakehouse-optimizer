# Streaming Lakehouse with a Learned Layout Optimizer

A fleet telemetry pipeline — Postgres → Debezium → Kafka → Flink → Apache Iceberg — where the physical table layout is chosen by a learned cost model instead of a static cron job, and every proposed change has to prove itself on a clone of the real table before anything touches production.

---

## The problem

Every Flink checkpoint flushes one file to Iceberg. At a 30-second checkpoint interval and 10,000 events per second, you accumulate roughly 2,880 files per day before any compaction runs. Scanning a table with thousands of tiny files is slow for a specific reason: every file carries a fixed planning and open overhead regardless of how much data it holds. The usual fix is a compaction cron job that merges small files into larger ones on a schedule.

That works until your workload changes. If ingest doubles, or queries shift from full scans to narrow time-range lookups, or a different partition granularity would cut scan time in half, the cron job has no way to notice. You find out months later when someone benchmarks the table.

The right question is: given what this table's queries actually look like right now, what file size, compaction schedule, and partition strategy produces the best scan latency at acceptable storage and write cost? That answer depends on the current workload, not a clock.

A learned cost model can approximate that function. A cron job cannot.

---

## Why a cost model instead of a direct policy

The obvious approach is to train a model that outputs the optimal layout. That does not work for two reasons.

There are no ground-truth "optimal layout" labels in production logs. You only observe outcomes of layouts you actually ran. A model trained to predict the optimal layout would just learn to replicate past decisions, inheriting whatever heuristic produced them.

Instead, this project learns a cost model: a function from (layout, workload) to (p95 scan latency, storage cost in dollars, write amplification). Once you have a cost model, you score every candidate layout and pick the best one. The model can be wrong. The shadow evaluation catches that. This is the same structure as learned query optimizers in the research literature, where models learn to rank plans rather than emit a single answer, specifically because ranking is safe to be wrong about.

---

## Quickstart

```bash
pip install -r requirements.txt
make test    # 50 tests, no cluster needed
make demo    # train the model, stream 300k rows, run one guarded optimization cycle
```

The demo output looks like this:

```
[1/3] training cost model on physical simulator ...
      held-out latency R² = 0.977
[2/3] streaming telemetry into a badly-laid-out table ...
      ingested 300000 rows -> 48 files, avg 1.3MB, p95 210.9ms
[3/3] running guarded optimize loop ...
      proposed : Layout(target_file_mb=512, compaction_trigger_files=100, partition_granularity='hour')
      promoted : True
      reason   : promoted: objective improved with no guarded regression
      after    : p95 82.9ms, 16 files
      ledger   : artifacts/promotion_ledger.jsonl
```

The table goes from 48 small files averaging 1.3MB to 16 properly-sized files, and p95 scan latency drops from 211ms to 83ms. The promotion decision and all supporting metrics are written to `artifacts/promotion_ledger.jsonl`.

### What the demo actually runs through

**Training.** `optimizer/train.py` drives the physical simulator through 40 workload scenarios across the full candidate grid: 4 file sizes × 3 compaction triggers × 3 partition strategies = 36 layout candidates per workload. For each combination it builds the table to a steady state, measures p95 latency, storage cost, and write amplification, then fits three independent gradient boosting regressors. The hold-out R² gate is 0.70. Training fails if the model does not clear it.

**Ingestion.** The demo creates a simulated Iceberg table with a deliberately bad incumbent (64MB target files, 100-file compaction trigger, device-bucket partitioning) and replays 300,000 telemetry rows through micro-batch ingest, producing 48 small files.

**Guarded optimization.** `orchestrator.run_once` scores every candidate layout with the cost model, then shadow-evaluates the top 3 on a deep copy of the table. The regression guard compares each shadow result against the baseline. The candidate must improve the weighted objective by at least 5% and must not regress latency, storage cost, or write amplification beyond their configured tolerances. The first candidate that clears all three checks is applied. If nothing clears the guard, the table is left alone and that decision is also written to the ledger.

---

## Running the full stack

```bash
make up            # starts Postgres, Kafka, Debezium, MinIO, Iceberg REST, Flink, Trino
make seed          # creates the source table in Postgres
make connect       # registers the Debezium CDC connector
make flink-submit  # submits the Flink job: Kafka → Iceberg with exactly-once semantics
make stream RPS=10000 SECONDS=120
```

After that you can query the Iceberg table from Trino at `localhost:8080`. The Airflow DAG in `airflow/dags/lakehouse_maintenance_dag.py` runs the guarded loop on a 30-minute schedule.

---

## How the guard works

The regression guard in `maintenance/regression_guard.py` enforces three conditions before any layout change is applied.

The weighted objective must improve by at least `min_improvement_pct` (default 5%). The objective is `w_latency * p95_ms + w_cost * storage_cost + w_write_amp * write_amplification`. This is the same formula used to train the cost model, so the thing the model was trained to predict is the thing the guard measures improvement on.

Latency must not get worse at all. A candidate that cuts storage cost by 30% but adds 10% to p95 scan latency is rejected outright, no exceptions.

Storage cost can regress by up to 10%. Write amplification has a separate absolute tolerance that should be set to a multiple of the table's actual size in production.

Every decision — promotion or rejection — is written to the append-only JSONL ledger with the full input metrics, the reason string, and a timestamp. The model never silently makes the table worse.

---

## Schema evolution

`catalog/schema_evolution.py` encodes Iceberg's backward-compatible evolution rules so a schema change can be validated in CI before it touches the table. Adding a nullable column, widening `int` to `long` or `float` to `double`, and increasing decimal precision are allowed. Dropping a column, narrowing a type, making a nullable column required, and silent renames are all rejected with an explicit error message that names the offending field.

---

## Exactly-once semantics

`ingest/exactly_once.py` models the two-phase commit invariant that Flink's Iceberg sink relies on. Each checkpoint is staged before it is visible, committed once the Flink barrier completes, and idempotent on replay after recovery. Aborting a checkpoint discards its staged rows without leaving a committed marker, so if the job recovers and re-stages the same checkpoint, it commits normally rather than being blocked by a phantom entry.

---

## Project layout

```
src/lakehouse/
  ingest/         telemetry generator, PyFlink job, exactly-once commit model
  optimizer/      feature contract, GBM cost model, layout search, training pipeline
  maintenance/    benchmark suite, shadow-eval, regression guard, orchestrator, Iceberg ops
  catalog/        schema evolution gate
  metrics/        Iceberg table stats and the deterministic physical simulator

infra/            docker-compose services
airflow/dags/     scheduled maintenance DAG
tests/            50 tests
docs/             architecture and ADRs
```

---

## What the tests cover

The cost model recovers a genuine physical signal from noisy labels (held-out R² gate). The layout search steers a tiny-file table toward larger files. The guard rejects a cheaper-but-slower candidate, a candidate that blows up write amplification, and a candidate where any metric is NaN (which would otherwise pass every float comparison silently). Shadow evaluation never mutates the production table. Compaction preserves row counts exactly. Checkpoint replay produces no duplicates. The abort-then-recovery path delivers rows correctly rather than being blocked by a phantom committed-checkpoint marker. Snapshot expiration keeps the metadata count bounded. Unsafe schema changes are rejected.

---

## Honest scope

The optimizer loop, regression guard, and all correctness invariants are real and fully tested against the physical simulator. The distributed runtime is wired in docker-compose with real Flink, Iceberg REST, and Trino configuration. The `PyIcebergBackend` apply path and Airflow task bodies are written for integration review rather than CI. A full code review of the project, including bugs found and fixed, production gaps, and a prioritized upgrade list, is in `REVIEW.md`.
