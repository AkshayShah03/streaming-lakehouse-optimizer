"""Regression guard + shadow evaluation: the safety core."""
import json

import pytest

from lakehouse.maintenance.regression_guard import (
    GuardConfig, evaluate_promotion, append_ledger)
from lakehouse.maintenance.shadow_eval import (
    EvalResult, baseline_result, shadow_evaluate)
from lakehouse.metrics.table_stats import Layout, SimulatedIcebergTable


def _er(latency, cost=1.0, wamp=10.0, rows=1000):
    return EvalResult(layout=Layout(128, 50, "day"), p95_latency_ms=latency,
                      storage_cost=cost, write_amplification=wamp, rows=rows)


def test_strictly_better_is_promoted():
    base = _er(latency=200, cost=2.0)
    cand = _er(latency=120, cost=1.5)
    d = evaluate_promotion(base, cand)
    assert d.promote and "promoted" in d.reason


def test_latency_regression_is_rejected_even_if_cheaper():
    # cheaper storage but slower queries -> must reject (latency tol = 0%)
    base = _er(latency=100, cost=5.0)
    cand = _er(latency=130, cost=1.0)
    d = evaluate_promotion(base, cand)
    assert not d.promote and "latency regressed" in d.reason


def test_marginal_improvement_below_threshold_rejected():
    base = _er(latency=100)
    cand = _er(latency=98)  # 2% < 5% threshold
    d = evaluate_promotion(base, cand, GuardConfig(min_improvement_pct=5.0))
    assert not d.promote and "below threshold" in d.reason


def test_ledger_is_append_only_and_jsonl(tmp_path):
    path = tmp_path / "ledger.jsonl"
    append_ledger(evaluate_promotion(_er(200), _er(120)), path)
    append_ledger(evaluate_promotion(_er(100), _er(130)), path)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    recs = [json.loads(l) for l in lines]
    assert recs[0]["promote"] is True
    assert recs[1]["promote"] is False


def test_shadow_eval_does_not_mutate_production():
    prod = SimulatedIcebergTable(layout=Layout(64, 100, "device_bucket"))
    for i in range(120):
        prod.ingest_micro_batch(mb=1.0, rows=5000, partition=f"b{i % 16}")
    files_before = prod.stats().file_count
    rows_before = prod.stats().rows

    shadow_evaluate(prod, Layout(256, 50, "day"))

    assert prod.stats().file_count == files_before  # untouched
    assert prod.stats().rows == rows_before


def test_shadow_eval_preserves_row_count():
    prod = SimulatedIcebergTable(layout=Layout(64, 200, "day"))
    for i in range(100):
        prod.ingest_micro_batch(mb=0.7, rows=3333, partition=f"d{i % 4}")
    rows_before = prod.stats().rows
    res = shadow_evaluate(prod, Layout(256, 50, "day"))
    assert res.rows == rows_before


def test_end_to_end_guard_promotes_real_improvement():
    """Tiny-file table -> compaction to 256MB should clear the guard."""
    prod = SimulatedIcebergTable(layout=Layout(64, 1000, "day"))
    for i in range(300):
        prod.ingest_micro_batch(mb=0.5, rows=2500, partition=f"d{i % 4}")
    base = baseline_result(prod)
    cand = shadow_evaluate(prod, Layout(256, 50, "day"))
    d = evaluate_promotion(base, cand)
    assert d.promote
    assert cand.p95_latency_ms < base.p95_latency_ms
