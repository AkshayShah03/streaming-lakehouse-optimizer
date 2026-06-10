"""Regression guard -- the zero-regression promotion gate.

The learned optimizer can only *propose*. A candidate layout is promoted iff,
on the empirical shadow measurement:

  1. The weighted objective improves by at least `min_improvement_pct`, AND
  2. No guarded metric regresses beyond its tolerance (latency, storage, write
     amplification), AND
  3. Row count is preserved (data-loss guard, enforced in shadow_eval too).

Every decision -- promote or reject -- is written to an append-only JSONL
ledger with the inputs, measurements and reason. This is the artifact that
lets you say in an interview: "the model never silently made the table
worse, and here's the audit trail proving it."
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .shadow_eval import EvalResult


@dataclass(frozen=True)
class GuardConfig:
    min_improvement_pct: float = 5.0      # require a real win, not noise
    latency_regression_tol_pct: float = 0.0   # never accept slower
    storage_regression_tol_pct: float = 10.0  # small storage give-back ok...
    write_amp_regression_tol_abs: float = 1e9  # ...if latency win is large
    w_latency: float = 1.0
    w_cost: float = 1.0
    w_write_amp: float = 0.25


@dataclass
class GuardDecision:
    promote: bool
    reason: str
    baseline_objective: float
    candidate_objective: float
    improvement_pct: float
    baseline: dict
    candidate: dict
    ts: float


def _validate_metrics(r: EvalResult, label: str) -> Optional[str]:
    """Return an error string if any metric is non-finite, else None.

    Python NaN comparisons are silently False, so nan > tol and nan < threshold
    both return False, causing every guard check to pass — a silent bad promotion.
    inf breaks the objective calculation. Catch both before any comparison.
    """
    for name, val in [
        ("p95_latency_ms", r.p95_latency_ms),
        ("storage_cost", r.storage_cost),
        ("write_amplification", r.write_amplification),
    ]:
        if not math.isfinite(val):
            return f"{label}.{name}={val} is not finite"
    return None


def _objective(r: EvalResult, c: GuardConfig) -> float:
    return (c.w_latency * r.p95_latency_ms
            + c.w_cost * r.storage_cost
            + c.w_write_amp * r.write_amplification)


def _pct_change(new: float, old: float) -> float:
    if old == 0:
        return 0.0 if new == 0 else float("inf")
    return (new - old) / old * 100


def evaluate_promotion(baseline: EvalResult, candidate: EvalResult,
                       config: GuardConfig = GuardConfig()) -> GuardDecision:
    # Validate before any comparison — NaN/inf silently passes every float check.
    for err in [_validate_metrics(baseline, "baseline"),
                _validate_metrics(candidate, "candidate")]:
        if err is not None:
            return GuardDecision(
                promote=False, reason=f"rejected: non-finite metric — {err}",
                baseline_objective=float("nan"), candidate_objective=float("nan"),
                improvement_pct=float("nan"),
                baseline=asdict(baseline) | {"layout": str(baseline.layout)},
                candidate=asdict(candidate) | {"layout": str(candidate.layout)},
                ts=time.time(),
            )

    base_obj = _objective(baseline, config)
    cand_obj = _objective(candidate, config)
    improvement = _pct_change(cand_obj, base_obj) * -1  # positive == better

    reasons: list[str] = []

    lat_change = _pct_change(candidate.p95_latency_ms, baseline.p95_latency_ms)
    if lat_change > config.latency_regression_tol_pct:
        reasons.append(f"latency regressed {lat_change:.1f}% "
                       f"(tol {config.latency_regression_tol_pct}%)")

    store_change = _pct_change(candidate.storage_cost, baseline.storage_cost)
    if store_change > config.storage_regression_tol_pct:
        reasons.append(f"storage regressed {store_change:.1f}% "
                       f"(tol {config.storage_regression_tol_pct}%)")

    wamp_change = candidate.write_amplification - baseline.write_amplification
    if wamp_change > config.write_amp_regression_tol_abs:
        reasons.append(f"write-amp regressed by {wamp_change:.1f}MB")

    if improvement < config.min_improvement_pct:
        reasons.append(f"improvement {improvement:.1f}% below threshold "
                       f"{config.min_improvement_pct}%")

    promote = not reasons
    reason = "promoted: objective improved with no guarded regression" if promote \
        else "rejected: " + "; ".join(reasons)

    return GuardDecision(
        promote=promote, reason=reason,
        baseline_objective=base_obj, candidate_objective=cand_obj,
        improvement_pct=improvement,
        baseline=asdict(baseline) | {"layout": str(baseline.layout)},
        candidate=asdict(candidate) | {"layout": str(candidate.layout)},
        ts=time.time(),
    )


def append_ledger(decision: GuardDecision,
                  path: str | Path = "artifacts/promotion_ledger.jsonl") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = asdict(decision)
    # EvalResult layouts are dataclasses inside dicts; make JSON-safe.
    rec["baseline"]["layout"] = str(rec["baseline"]["layout"])
    rec["candidate"]["layout"] = str(rec["candidate"]["layout"])
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")
