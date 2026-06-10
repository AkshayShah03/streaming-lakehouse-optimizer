"""Search the layout space using the learned cost model.

Enumerate a bounded candidate grid, score each with the cost model, and
return the best together with the predicted improvement over the incumbent.
The search itself is cheap and exhaustive over a curated grid -- the value
is in the *learned* scoring, not in fancy search. Returning a ranked list
(not just the argmin) mirrors learning-to-rank optimizers and lets the
shadow-eval stage try the top-k if the best fails validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Optional

from .cost_model import CostModel, CostPrediction
from .features import WorkloadFeatures
from ..metrics.table_stats import Layout


# Curated grid -- bounded so search is deterministic and explainable.
FILE_SIZES_MB = (64, 128, 256, 512)
TRIGGERS = (20, 50, 100)
PARTITIONS = ("hour", "day", "device_bucket")


@dataclass
class ScoredLayout:
    layout: Layout
    prediction: CostPrediction
    objective: float


@dataclass
class Recommendation:
    incumbent: ScoredLayout
    best: ScoredLayout
    ranked: list[ScoredLayout]
    predicted_improvement_pct: float

    @property
    def changes_layout(self) -> bool:
        return self.best.layout != self.incumbent.layout


def candidate_layouts() -> list[Layout]:
    out = []
    for fs, tr, part in product(FILE_SIZES_MB, TRIGGERS, PARTITIONS):
        lay = Layout(target_file_mb=fs, compaction_trigger_files=tr,
                     partition_granularity=part)
        lay.validate()
        out.append(lay)
    return out


def recommend(model: CostModel, incumbent: Layout, wf: WorkloadFeatures,
              *, w_latency: float = 1.0, w_cost: float = 1.0,
              w_write_amp: float = 0.25) -> Recommendation:
    def score(lay: Layout) -> ScoredLayout:
        pred = model.predict(lay, wf)
        return ScoredLayout(lay, pred,
                            pred.objective(w_latency, w_cost, w_write_amp))

    ranked = sorted((score(l) for l in candidate_layouts()),
                    key=lambda s: s.objective)
    inc = score(incumbent)
    best = ranked[0]
    improvement = (inc.objective - best.objective) / inc.objective * 100 \
        if inc.objective else 0.0
    return Recommendation(incumbent=inc, best=best, ranked=ranked,
                          predicted_improvement_pct=improvement)
