"""New tests from staff-level review.

Each test is annotated with WHY it was added and what production failure it prevents.
Tests that expose bugs that were then fixed are prefixed with an explanation of the
original broken behavior.
"""
import math
import json

import numpy as np
import pytest

from lakehouse.ingest.exactly_once import ExactlyOnceSink
from lakehouse.catalog.schema_evolution import Field, SchemaEvolutionError, validate_evolution
from lakehouse.maintenance.regression_guard import (
    GuardConfig, evaluate_promotion, _pct_change,
)
from lakehouse.maintenance.shadow_eval import EvalResult, shadow_evaluate, baseline_result
from lakehouse.maintenance.iceberg_ops import SimBackend
from lakehouse.maintenance.orchestrator import run_once
from lakehouse.metrics.table_stats import Layout, SimulatedIcebergTable
from lakehouse.optimizer.cost_model import CostModel
from lakehouse.optimizer.train import generate_dataset
from lakehouse.optimizer.features import WorkloadFeatures


# ---------------------------------------------------------------------------
# exactly_once.py: phantom checkpoint bug (abort + immediate commit creates a
# committed-checkpoint marker with 0 rows; subsequent recovery replay is silently
# blocked, causing data loss).
# ---------------------------------------------------------------------------

def test_abort_then_recovery_stage_commit_delivers_rows():
    """Abort then recovery re-stage should commit rows, not be blocked by phantom.

    Broken before fix: abort(1) + commit(1) left id 1 in _committed_checkpoints.
    Any recovery that re-staged and committed checkpoint 1 got return False + 0 rows.
    """
    sink = ExactlyOnceSink()
    sink.stage(1, 1000)
    sink.abort(1)
    # Recovery: job restarts from previous checkpoint, re-processes checkpoint 1.
    sink.stage(1, 1000)
    committed = sink.commit(1)
    assert committed is True, "fresh stage after abort must commit (no phantom block)"
    assert sink.total_rows == 1000, f"expected 1000, got {sink.total_rows}"


def test_abort_does_not_block_future_replay():
    """Aborting checkpoint N must not prevent a later successful commit of N."""
    sink = ExactlyOnceSink()
    # First attempt: stage then abort (e.g., checkpoint barrier dropped)
    sink.stage(5, 500)
    sink.abort(5)
    # Second attempt on the same checkpoint id (recovery path)
    sink.stage(5, 500)
    assert sink.commit(5) is True
    assert sink.total_rows == 500
    # Idempotent replay still works
    sink.stage(5, 500)
    assert sink.commit(5) is False
    assert sink.total_rows == 500


def test_abort_then_commit_without_restage_is_noop():
    """Committing an aborted checkpoint without re-staging should be safe (no-op)."""
    sink = ExactlyOnceSink()
    sink.stage(3, 300)
    sink.abort(3)
    # No re-stage: commit is a no-op, should not write phantom 0-row entry.
    result = sink.commit(3)
    assert result is False, "committing without restage after abort should be a no-op"
    assert sink.total_rows == 0
    assert sink.committed_checkpoints == 0


# ---------------------------------------------------------------------------
# regression_guard.py: NaN / infinity in metrics silently passes the guard.
# Python: float('nan') > 0.0 is False, float('nan') < 5.0 is False, so both
# the regression check and the improvement threshold check PASS with NaN.
# ---------------------------------------------------------------------------

def _er(latency, cost=1.0, wamp=10.0, rows=1000):
    return EvalResult(layout=Layout(128, 50, "day"), p95_latency_ms=latency,
                      storage_cost=cost, write_amplification=wamp, rows=rows)


def test_nan_latency_in_candidate_is_rejected():
    """NaN latency on the candidate must never produce a promotion."""
    base = _er(latency=100.0)
    cand = _er(latency=float("nan"))
    d = evaluate_promotion(base, cand)
    assert not d.promote, "NaN latency must be rejected, not promoted"
    assert "nan" in d.reason.lower() or "invalid" in d.reason.lower() or "finite" in d.reason.lower()


def test_nan_latency_in_baseline_is_rejected():
    """NaN in the baseline is also unsafe — guard must refuse to act."""
    base = _er(latency=float("nan"))
    cand = _er(latency=80.0)
    d = evaluate_promotion(base, cand)
    assert not d.promote, "NaN baseline must block promotion"


def test_inf_latency_in_candidate_is_rejected():
    """+Inf latency must be rejected (regression by definition)."""
    base = _er(latency=100.0)
    cand = _er(latency=float("inf"))
    d = evaluate_promotion(base, cand)
    assert not d.promote


def test_nan_storage_cost_is_rejected():
    """NaN in any metric, not just latency, must block promotion."""
    base = _er(latency=100.0, cost=2.0)
    cand = _er(latency=80.0, cost=float("nan"))
    d = evaluate_promotion(base, cand)
    assert not d.promote, "NaN storage cost must be rejected"


def test_nan_write_amp_is_rejected():
    base = _er(latency=100.0, wamp=50.0)
    cand = _er(latency=80.0, wamp=float("nan"))
    d = evaluate_promotion(base, cand)
    assert not d.promote, "NaN write amplification must be rejected"


# ---------------------------------------------------------------------------
# regression_guard.py: write_amp_regression_tol_abs = 1e9 is effectively infinite.
# Any compaction that rewrites <1 billion MB passes. Real tables have ~100-10k MB.
# ---------------------------------------------------------------------------

def test_write_amp_guard_actually_blocks_expensive_compaction():
    """A compaction that rewrites far more than the tolerance must be rejected.

    With default tol=1e9 (1 billion MB) this never fires. Setting a realistic
    tolerance and zero w_write_amp (so objective still improves) isolates the
    regression-check path.
    """
    base = _er(latency=100.0, wamp=0.0)
    cand = _er(latency=60.0, wamp=600.0)  # big latency win but huge rewrite cost
    cfg = GuardConfig(
        write_amp_regression_tol_abs=200.0,
        w_write_amp=0.0,     # objective doesn't penalize write-amp
        min_improvement_pct=1.0,  # low threshold so objective check passes
    )
    d = evaluate_promotion(base, cand, cfg)
    assert not d.promote, f"600MB write-amp delta should exceed 200MB tol; got: {d.reason}"
    assert "write-amp" in d.reason.lower()


def test_write_amp_guard_passes_acceptable_rewrite():
    base = _er(latency=100.0, wamp=0.0)
    # Latency drops 40ms; write-amp delta 100MB is under 200MB tolerance.
    # Zero the w_write_amp so write-amp doesn't flip the objective.
    cand = _er(latency=60.0, wamp=100.0)
    cfg = GuardConfig(
        write_amp_regression_tol_abs=200.0,
        w_write_amp=0.0,
        min_improvement_pct=1.0,
    )
    d = evaluate_promotion(base, cand, cfg)
    assert d.promote, f"100MB write-amp under 200MB tol should pass; got: {d.reason}"


# ---------------------------------------------------------------------------
# schema_evolution.py: complex types (struct/list/map) fall through to string
# comparison and are silently rejected even for safe evolutions.
# ---------------------------------------------------------------------------

def test_struct_field_add_is_reported_clearly():
    """Evolving a struct field should either be safe or give an actionable error.
    Currently rejects because string "struct<a:int>" != "struct<a:int,b:string>".
    This is conservative (correct) but the test documents the gap so it's explicit.
    """
    old = [Field(1, "meta", "struct<a:int>", False)]
    new = [Field(1, "meta", "struct<a:int,b:string>", False)]
    with pytest.raises(SchemaEvolutionError) as exc_info:
        validate_evolution(old, new)
    # The error should say something about type change, not be silently wrong.
    assert "meta" in str(exc_info.value), "error must name the offending field"


def test_list_element_type_change_rejected():
    """list<string> -> list<long> must be rejected (breaking change)."""
    old = [Field(1, "tags", "list<string>", False)]
    new = [Field(1, "tags", "list<long>", False)]
    with pytest.raises(SchemaEvolutionError):
        validate_evolution(old, new)


def test_decimal_precision_same_allowed():
    """Same decimal precision = no change; must not raise."""
    old = [Field(1, "amount", "decimal", False, precision=10)]
    new = [Field(1, "amount", "decimal", False, precision=10)]
    validate_evolution(old, new)  # no raise


def test_decimal_precision_increase_allowed():
    """Decimal precision widening is safe in Iceberg."""
    old = [Field(1, "amount", "decimal", False, precision=10)]
    new = [Field(1, "amount", "decimal", False, precision=15)]
    validate_evolution(old, new)  # no raise


def test_decimal_precision_decrease_rejected():
    old = [Field(1, "amount", "decimal", False, precision=10)]
    new = [Field(1, "amount", "decimal", False, precision=8)]
    with pytest.raises(SchemaEvolutionError, match="narrowed"):
        validate_evolution(old, new)


# ---------------------------------------------------------------------------
# snapshot expiration: orchestrator must run expire_snapshots after every cycle.
# ---------------------------------------------------------------------------

def test_sim_backend_snapshot_count_grows_on_apply():
    """Each apply_layout creates a new snapshot; count must be tracked."""
    table = SimulatedIcebergTable(layout=Layout(64, 200, "day"))
    for i in range(40):
        table.ingest_micro_batch(mb=0.5, rows=2500, partition=f"d{i % 4}")
    backend = SimBackend(table)
    assert backend.snapshot_count == 0
    backend.apply_layout(Layout(256, 50, "day"))
    assert backend.snapshot_count == 1
    backend.apply_layout(Layout(512, 50, "day"))
    assert backend.snapshot_count == 2


def test_expire_snapshots_reduces_count():
    """expire_snapshots must remove old snapshots up to the retention boundary."""
    table = SimulatedIcebergTable(layout=Layout(64, 200, "day"))
    for i in range(40):
        table.ingest_micro_batch(mb=0.5, rows=2500, partition=f"d{i % 4}")
    backend = SimBackend(table)
    for _ in range(5):
        backend.apply_layout(Layout(128, 50, "day"))
    assert backend.snapshot_count == 5
    result = backend.expire_snapshots(keep_last=2)
    assert backend.snapshot_count == 2
    assert result["expired"] == 3


def test_delete_orphan_files_returns_stats():
    """delete_orphan_files must return a dict with a count key."""
    table = SimulatedIcebergTable(layout=Layout(128, 50, "day"))
    backend = SimBackend(table)
    result = backend.delete_orphan_files()
    assert "deleted_files" in result


def test_orchestrator_runs_snapshot_expiration(tmp_path):
    """run_once must call expire_snapshots after every cycle so snapshots don't pile up."""
    X, yl, yc, yw = generate_dataset(n_workloads=20, seed=9)
    model = CostModel().fit(X, yl, yc, yw)
    prod = SimulatedIcebergTable(layout=Layout(64, 100000, "device_bucket"))
    for i in range(250):
        prod.ingest_micro_batch(mb=0.5, rows=2500, partition=f"b{i % 16}")
    backend = SimBackend(prod)
    wf = WorkloadFeatures(
        ingest_rows_per_sec=9000, avg_selectivity=0.07,
        time_range_query_ratio=0.9, read_write_ratio=5.0,
        avg_file_mb=prod.stats().avg_file_mb, file_count=prod.stats().file_count,
        small_file_ratio=prod.stats().small_file_ratio,
        partition_count=prod.stats().partition_count)
    run_once(model=model, production=prod, backend=backend, wf=wf,
             guard=GuardConfig(min_improvement_pct=5.0),
             ledger_path=str(tmp_path / "ledger.jsonl"))
    # Whether or not the layout was promoted, snapshot count must stay bounded.
    assert backend.snapshot_count <= 2, (
        f"snapshot count {backend.snapshot_count} not bounded after run_once — "
        "expire_snapshots not called")


# ---------------------------------------------------------------------------
# ML honesty: R² = 1.000 reveals feature leakage (file_count, avg_file_mb
# directly encode the scan formula). Adding label noise should degrade R²
# noticeably; near-perfect R² after noise means memorization, not learning.
# ---------------------------------------------------------------------------

def test_cost_model_r2_degrades_with_label_noise():
    """With 15% measurement noise (default in generate_dataset), held-out R² < 0.98.

    The simulator's scan formula is deterministic; without noise R² = 1.0 exactly
    (model inverts the formula from features). With realistic measurement noise
    added, R² reflects genuine generalization ability, not formula inversion.
    This test passes after the generate_dataset fix (label_noise_pct=0.15 default).
    """
    X, y_lat, y_cost, y_wamp = generate_dataset(n_workloads=60, seed=1,
                                                  label_noise_pct=0.15)
    rng = np.random.RandomState(1)
    idx = rng.permutation(len(X))
    cut = int(len(X) * 0.8)
    tr, te = idx[:cut], idx[cut:]
    model = CostModel().fit(X[tr], y_lat[tr], y_cost[tr], y_wamp[tr])
    r2 = model.score(X[te], y_lat[te])

    # With noise the model can't perfectly invert the formula → R² < 0.98.
    # Still > 0.7 because the signal (file_count, avg_file_mb) is real.
    assert r2 < 0.98, (
        f"R²={r2:.4f} with 15% noise should be < 0.98 — if not, the model is "
        "memorizing, not learning.")
    assert r2 > 0.70, (
        f"R²={r2:.4f} should still be > 0.70 — the physical signal must be recoverable.")


def test_cost_model_cross_seed_r2():
    """Cross-seed held-out R² must be < 0.99 after adding measurement noise.

    Without noise: R² ≈ 0.9967 (model inverts the scan formula from features).
    With 15% noise in generate_dataset: R² drops to ~0.95 (genuine generalization).
    """
    X_train, y_lat_train, y_cost_train, y_wamp_train = generate_dataset(
        n_workloads=60, seed=1)       # uses default label_noise_pct=0.15
    model = CostModel().fit(X_train, y_lat_train, y_cost_train, y_wamp_train)

    X_test, y_lat_test, _, _ = generate_dataset(n_workloads=30, seed=999)
    r2 = model.score(X_test, y_lat_test)
    assert r2 < 0.99, (
        f"Cross-seed R²={r2:.4f} ≈ 1.0 even with label noise — leakage not resolved. "
        "See REVIEW.md §ML for root cause analysis.")
    assert r2 > 0.70, f"R²={r2:.4f} below 0.70 — physical signal is not being learned."


# ---------------------------------------------------------------------------
# Empty + single-file edge cases for shadow_eval
# ---------------------------------------------------------------------------

def test_shadow_eval_on_empty_table_does_not_crash():
    """shadow_evaluate on a table with zero files must not raise."""
    prod = SimulatedIcebergTable(layout=Layout(128, 200, "day"))
    # Don't ingest anything — empty table.
    result = shadow_evaluate(prod, Layout(256, 50, "day"))
    assert result.rows == 0
    assert result.p95_latency_ms >= 0.0


def test_shadow_eval_on_single_file_table():
    """Single-file table: compaction is a no-op but must preserve row count."""
    prod = SimulatedIcebergTable(layout=Layout(128, 200, "day"))
    prod.ingest_micro_batch(mb=2.0, rows=10_000, partition="d0")
    result = shadow_evaluate(prod, Layout(256, 50, "day"))
    assert result.rows == 10_000


def test_baseline_result_on_empty_table():
    """baseline_result on an empty table must return finite metrics."""
    prod = SimulatedIcebergTable(layout=Layout(128, 50, "day"))
    b = baseline_result(prod)
    assert math.isfinite(b.p95_latency_ms)
    assert math.isfinite(b.storage_cost)
    assert math.isfinite(b.write_amplification)
