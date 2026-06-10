"""Learned cost model: (layout, workload) -> (p95 scan latency, storage $, write amp).

Design choice (defensible in interview): we do NOT regress directly to an
"optimal layout" -- there are no ground-truth optimal labels in production.
Instead we learn a *cost model* and search over candidate layouts, exactly
the structure of learned query optimizers (Lero learns to rank plans; we
learn to score layouts). The model is trained on observed
(layout, workload, measured-outcome) tuples collected from the running
system; here we bootstrap from the physical simulator and then the
production loop continually appends real measurements.

Three independent regressors keep the targets interpretable and let the
regression guard reason per-metric (a layout may cut latency but blow up
write amplification -- we must see both).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from .features import WorkloadFeatures
from ..metrics.table_stats import Layout


_PARTITION_ENCODING = {"hour": 0.0, "day": 1.0, "device_bucket": 2.0}


def encode(layout: Layout, wf: WorkloadFeatures) -> list[float]:
    """Joint feature vector for the cost model."""
    return [
        float(layout.target_file_mb),
        float(layout.compaction_trigger_files),
        _PARTITION_ENCODING[layout.partition_granularity],
        *wf.to_vector(),
    ]


@dataclass
class CostPrediction:
    p95_latency_ms: float
    storage_cost: float
    write_amplification: float

    def objective(self, w_latency: float = 1.0, w_cost: float = 1.0,
                  w_write_amp: float = 0.25) -> float:
        """Lower is better. Weights are explicit + tunable per SLA."""
        return (w_latency * self.p95_latency_ms
                + w_cost * self.storage_cost
                + w_write_amp * self.write_amplification)


class CostModel:
    def __init__(self) -> None:
        mk = lambda: GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0)
        self._latency = mk()
        self._cost = mk()
        self._wamp = mk()
        self._fitted = False

    def fit(self, X: np.ndarray, y_latency: np.ndarray,
            y_cost: np.ndarray, y_wamp: np.ndarray) -> "CostModel":
        self._latency.fit(X, y_latency)
        self._cost.fit(X, y_cost)
        self._wamp.fit(X, y_wamp)
        self._fitted = True
        return self

    def predict(self, layout: Layout, wf: WorkloadFeatures) -> CostPrediction:
        if not self._fitted:
            raise RuntimeError("CostModel.predict called before fit/load")
        x = np.array([encode(layout, wf)])
        return CostPrediction(
            p95_latency_ms=float(self._latency.predict(x)[0]),
            storage_cost=float(self._cost.predict(x)[0]),
            write_amplification=float(self._wamp.predict(x)[0]),
        )

    def score(self, X: np.ndarray, y_latency: np.ndarray) -> float:
        """R^2 of the latency head -- the dominant target -- for sanity gating."""
        return float(self._latency.score(X, y_latency))

    # -- persistence: tie a model version to the feature contract ----------
    def save(self, path: str | Path) -> None:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"latency": self._latency, "cost": self._cost, "wamp": self._wamp,
             "features": WorkloadFeatures.feature_names()}, path)

    @classmethod
    def load(cls, path: str | Path) -> "CostModel":
        import joblib
        blob = joblib.load(path)
        if blob["features"] != WorkloadFeatures.feature_names():
            raise ValueError("feature contract drift: retrain before serving")
        m = cls()
        m._latency, m._cost, m._wamp = blob["latency"], blob["cost"], blob["wamp"]
        m._fitted = True
        return m
