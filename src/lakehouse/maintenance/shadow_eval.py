"""Shadow evaluation.

Before any layout change is promoted, we apply it to a *clone* of the
production table state and measure actual benchmark outcomes -- never trust
the model's prediction alone. This is the empirical check that turns a
learned recommendation into a safe action: the model proposes, the shadow
measures, the guard decides.

Invariant: shadow evaluation MUST NOT mutate the production table. The clone
is the unit of mutation; production is read-only here.
"""
from __future__ import annotations

from dataclasses import dataclass

from .benchmark import measure_p95_latency
from ..metrics.table_stats import Layout, SimulatedIcebergTable


@dataclass(frozen=True)
class EvalResult:
    layout: Layout
    p95_latency_ms: float
    storage_cost: float
    write_amplification: float
    rows: int


def shadow_evaluate(production: SimulatedIcebergTable,
                    candidate: Layout) -> EvalResult:
    """Clone production, apply candidate compaction, measure. Pure wrt production."""
    rows_before = production.stats().rows
    clone = production.clone()
    clone.layout = candidate
    clone.rewrite_data_files(candidate.target_file_mb)

    result = EvalResult(
        layout=candidate,
        p95_latency_ms=measure_p95_latency(clone),
        storage_cost=clone.storage_cost_units(),
        write_amplification=clone.bytes_rewritten_mb,
        rows=clone.stats().rows,
    )
    # Backfill/data-loss guard: compaction must preserve row count exactly.
    if result.rows != rows_before:
        raise AssertionError(
            f"shadow eval row mismatch: {rows_before} -> {result.rows}")
    return result


def baseline_result(production: SimulatedIcebergTable) -> EvalResult:
    """Measure the incumbent layout as-is (no rewrite) for comparison."""
    st = production.stats()
    return EvalResult(
        layout=production.layout,
        p95_latency_ms=measure_p95_latency(production),
        storage_cost=production.storage_cost_units(),
        write_amplification=production.bytes_rewritten_mb,
        rows=st.rows,
    )
