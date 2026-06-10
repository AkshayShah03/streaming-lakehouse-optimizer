"""Canonical benchmark suite + p95 measurement.

Used by BOTH training-data generation and shadow evaluation, so the metric
the regression guard promotes on is exactly the metric the cost model was
trained to predict. Against a real cluster, `measure_p95_latency` is swapped
for a Trino/Spark runner reading wall-clock from the query log; the simulator
implements the same `.scan()` interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class BenchmarkQuery:
    name: str
    selectivity: float        # fraction of bytes read
    is_time_range: bool       # benefits from time-based partition pruning
    weight: int               # relative frequency in the workload


class Scannable(Protocol):
    def scan(self, selectivity: float, partition_pruned: bool = True) -> float: ...


# Mostly selective time-range dashboards + a few wide scans (telemetry shape).
DEFAULT_SUITE: list[BenchmarkQuery] = [
    BenchmarkQuery("recent_telemetry", 0.02, True, 5),
    BenchmarkQuery("device_day", 0.05, True, 3),
    BenchmarkQuery("fleet_rollup", 0.30, True, 2),
    BenchmarkQuery("adhoc_full_scan", 0.90, False, 1),
]


def measure_p95_latency(table: Scannable,
                        suite: list[BenchmarkQuery] | None = None,
                        repeats: int = 20) -> float:
    """Weighted p95 latency over the suite. Deterministic given the table."""
    suite = suite or DEFAULT_SUITE
    samples: list[float] = []
    for q in suite:
        lat = table.scan(q.selectivity, partition_pruned=q.is_time_range)
        samples.extend([lat] * (q.weight * repeats))
    return float(np.percentile(samples, 95))
