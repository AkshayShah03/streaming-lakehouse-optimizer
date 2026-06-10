# ADR 0001 — Learn a cost model + search, do not regress to "optimal layout"

Status: Accepted

## Context
We want to replace static, cron-based Iceberg compaction with a workload-aware
decision. The naive framing is supervised: train a model to output the optimal
layout. That framing is wrong for two reasons:

1. **No labels.** There is no ground-truth "optimal layout" per workload in
   production logs; we only observe outcomes of layouts we actually ran.
2. **Distribution shift.** A direct policy trained on historical choices
   inherits whatever heuristic produced those choices.

## Decision
Learn a **cost model** `f(layout, workload) -> (p95_latency, storage_$,
write_amplification)` from observed `(layout, workload, measured-outcome)`
tuples, then **search** a bounded, curated layout grid and score each candidate.
Return a full ranking, not just the argmin, so downstream stages can try the
top-k.

This mirrors the learned-query-optimizer literature (e.g. Lero learns to *rank*
plans rather than emit one) and the broader ML-for-systems pattern of replacing
hand-tuned heuristics with a model trained on workload signals — while keeping
the action space explicit and auditable.

## Consequences
- The model can be wrong without being dangerous: it only *proposes*. The
  empirical shadow-evaluation + regression guard are the authority on what ships.
- Three separate regressor heads (latency / storage / write-amp) keep the guard
  able to reason per-metric, so a candidate that wins latency but blows up write
  amplification is visible and rejectable.
- The same benchmark suite generates training labels and drives shadow eval, so
  the promoted metric is the trained metric — no train/serve objective drift.

## Alternatives considered
- **Direct layout policy (rejected):** no labels, inherits heuristic bias.
- **Reinforcement learning (deferred):** higher ceiling, but needs far more
  online interaction and a safety story we already get more cheaply from the
  shadow-eval guard. Revisit once the cost-model loop is saturated.
