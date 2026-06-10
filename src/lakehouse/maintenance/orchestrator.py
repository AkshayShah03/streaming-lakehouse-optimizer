"""The maintenance control loop.

  observe workload  ->  model proposes layout  ->  shadow-evaluate top-k
  ->  regression guard decides  ->  apply iff promoted  ->  log decision

This is the orchestration the Airflow DAG calls on a schedule. It is the
"closed loop" that replaces static cron compaction with a guarded,
workload-aware decision -- and crucially it can DECLINE to act.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..metrics.table_stats import Layout, SimulatedIcebergTable
from ..optimizer.cost_model import CostModel
from ..optimizer.features import WorkloadFeatures
from ..optimizer.layout_search import recommend
from .iceberg_ops import MaintenanceBackend
from .regression_guard import GuardConfig, append_ledger, evaluate_promotion
from .shadow_eval import baseline_result, shadow_evaluate


@dataclass
class LoopResult:
    proposed: Layout
    promoted: bool
    reason: str
    applied: Optional[dict]


def run_once(*, model: CostModel, production: SimulatedIcebergTable,
             backend: MaintenanceBackend, wf: WorkloadFeatures,
             guard: GuardConfig = GuardConfig(), top_k: int = 3,
             ledger_path: str = "artifacts/promotion_ledger.jsonl",
             snapshot_keep_last: int = 10) -> LoopResult:
    rec = recommend(model, production.layout, wf,
                    w_latency=guard.w_latency, w_cost=guard.w_cost,
                    w_write_amp=guard.w_write_amp)

    base = baseline_result(production)

    # Try the top-k model proposals; promote the first that clears the guard.
    for scored in rec.ranked[:top_k]:
        if scored.layout == production.layout:
            continue
        cand = shadow_evaluate(production, scored.layout)
        decision = evaluate_promotion(base, cand, guard)
        append_ledger(decision, ledger_path)
        if decision.promote:
            applied = backend.apply_layout(scored.layout)
            # Always run housekeeping regardless of whether promotion happened.
            # Streaming ingest creates one snapshot per checkpoint (every 30s);
            # without expiration, metadata load time grows unbounded within days.
            backend.expire_snapshots(keep_last=snapshot_keep_last)
            backend.delete_orphan_files()
            return LoopResult(proposed=scored.layout, promoted=True,
                              reason=decision.reason, applied=applied)

    # Even on a no-promote cycle, run housekeeping so snapshots don't pile up.
    backend.expire_snapshots(keep_last=snapshot_keep_last)
    backend.delete_orphan_files()
    return LoopResult(proposed=rec.best.layout, promoted=False,
                      reason="no candidate cleared the regression guard",
                      applied=None)
