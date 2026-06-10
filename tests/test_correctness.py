"""Correctness invariants: backfill, schema evolution, exactly-once, e2e loop."""
import pytest

from lakehouse.catalog.schema_evolution import (
    Field, SchemaEvolutionError, is_compatible, validate_evolution)
from lakehouse.ingest.exactly_once import ExactlyOnceSink
from lakehouse.ingest.telemetry_generator import generate
from lakehouse.maintenance.iceberg_ops import SimBackend
from lakehouse.maintenance.orchestrator import run_once
from lakehouse.maintenance.regression_guard import GuardConfig
from lakehouse.metrics.table_stats import Layout, SimulatedIcebergTable
from lakehouse.optimizer.cost_model import CostModel
from lakehouse.optimizer.features import WorkloadFeatures
from lakehouse.optimizer.train import generate_dataset


# ---- backfill / compaction correctness -----------------------------------

def test_compaction_preserves_rows_and_reduces_files():
    t = SimulatedIcebergTable(layout=Layout(64, 100000, "day"))
    total = 0
    for i in range(200):
        t.ingest_micro_batch(mb=0.6, rows=3000, partition=f"d{i % 5}")
        total += 3000
    before = t.stats()
    t.rewrite_data_files(target_file_mb=256)
    after = t.stats()
    assert after.rows == total          # zero data loss
    assert after.file_count < before.file_count  # actually compacted


def test_compaction_is_partition_local():
    t = SimulatedIcebergTable(layout=Layout(128, 100000, "day"))
    for i in range(40):
        t.ingest_micro_batch(mb=2.0, rows=10000, partition=f"d{i % 4}")
    t.rewrite_data_files(256)
    assert t.stats().partition_count == 4  # partitions never merged


# ---- schema evolution -----------------------------------------------------

OLD = [
    Field(1, "vehicle_id", "long", True),
    Field(2, "speed_kph", "float", False),
]


def test_add_optional_column_is_compatible():
    new = OLD + [Field(3, "tire_psi", "double", False)]
    validate_evolution(OLD, new)  # no raise


def test_add_required_column_rejected():
    new = OLD + [Field(3, "tire_psi", "double", True)]
    assert not is_compatible(OLD, new)


def test_type_widening_allowed_narrowing_rejected():
    widen = [Field(1, "vehicle_id", "long", True),
             Field(2, "speed_kph", "double", False)]  # float->double ok
    validate_evolution(OLD, widen)
    narrow = [Field(1, "vehicle_id", "int", True),  # long->int unsafe
              Field(2, "speed_kph", "float", False)]
    with pytest.raises(SchemaEvolutionError):
        validate_evolution(OLD, narrow)


def test_column_drop_rejected():
    new = [Field(1, "vehicle_id", "long", True)]
    with pytest.raises(SchemaEvolutionError):
        validate_evolution(OLD, new)


def test_optional_to_required_rejected():
    new = [Field(1, "vehicle_id", "long", True),
           Field(2, "speed_kph", "float", True)]
    assert not is_compatible(OLD, new)


# ---- exactly-once ---------------------------------------------------------

def test_checkpoint_replay_does_not_duplicate():
    sink = ExactlyOnceSink()
    sink.stage(1, 1000)
    assert sink.commit(1) is True
    # recovery replays checkpoint 1
    sink.stage(1, 1000)
    assert sink.commit(1) is False     # idempotent no-op
    assert sink.total_rows == 1000     # not 2000


def test_aborted_checkpoint_writes_no_rows():
    sink = ExactlyOnceSink()
    sink.stage(5, 500)
    sink.abort(5)
    # After abort, commit without re-stage is a safe no-op (returns False).
    # Previously returned True and added a phantom committed marker, blocking
    # legitimate recovery re-plays of the same checkpoint id.
    assert sink.commit(5) is False
    assert sink.total_rows == 0


def test_distinct_checkpoints_accumulate():
    sink = ExactlyOnceSink()
    for cp in range(1, 6):
        sink.stage(cp, 100)
        sink.commit(cp)
    assert sink.committed_checkpoints == 5
    assert sink.total_rows == 500


# ---- telemetry generator --------------------------------------------------

def test_generator_hits_requested_rate():
    total = sum(len(b) for b in generate(rps=10_000, seconds=2, fleet_size=500))
    # 10 batches/sec * 2s = 20 batches of 1000 -> ~20k rows
    assert 18_000 <= total <= 22_000


def test_generator_fields_in_range():
    batch = next(iter(generate(rps=1000, seconds=1)))
    r = batch[0]
    assert r.battery_pct <= 100 and r.speed_kph >= 0
    assert -180 <= r.lon <= 180 and -90 <= r.lat <= 90


# ---- orchestrator end-to-end ---------------------------------------------

def test_orchestrator_promotes_or_declines_safely():
    X, yl, yc, yw = generate_dataset(n_workloads=20, seed=9)
    model = CostModel().fit(X, yl, yc, yw)

    prod = SimulatedIcebergTable(layout=Layout(64, 100000, "device_bucket"))
    for i in range(250):
        prod.ingest_micro_batch(mb=0.5, rows=2500, partition=f"b{i % 16}")
    st = prod.stats()
    wf = WorkloadFeatures(
        ingest_rows_per_sec=9000, avg_selectivity=0.07,
        time_range_query_ratio=0.9, read_write_ratio=5.0,
        avg_file_mb=st.avg_file_mb, file_count=st.file_count,
        small_file_ratio=st.small_file_ratio, partition_count=st.partition_count)

    rows_before = st.rows
    res = run_once(model=model, production=prod, backend=SimBackend(prod),
                   wf=wf, guard=GuardConfig(min_improvement_pct=5.0),
                   ledger_path="artifacts/test_ledger.jsonl")
    # Either it promoted a genuine win, or it safely declined -- never unsafe.
    assert isinstance(res.promoted, bool)
    assert prod.stats().rows == rows_before  # invariant holds regardless
    if res.promoted:
        assert res.applied is not None
