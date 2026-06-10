"""Generate (layout, workload) -> measured-outcome training data and fit the
cost model.

Bootstrapping: we drive the physical simulator to a steady state under each
candidate layout + sampled workload, then *measure* p95 latency, storage
cost and write amplification. In production this same table is fed by the
real measurement stream appended by the maintenance loop, so the model is
continually retrained on observed outcomes rather than the simulator.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from .cost_model import CostModel, encode
from .features import WorkloadFeatures
from .layout_search import candidate_layouts
from ..maintenance.benchmark import measure_p95_latency
from ..metrics.table_stats import Layout, SimulatedIcebergTable


def _build_to_steady_state(layout: Layout, ingest_rps: float,
                           rng: random.Random) -> SimulatedIcebergTable:
    """Stream micro-batches for a simulated hour at the given ingest rate."""
    table = SimulatedIcebergTable(layout=layout, seed=rng.randint(0, 10_000))
    # one micro-batch per streaming checkpoint (~30s); size scales with rps
    checkpoints = 120
    rows_per_cp = max(1, int(ingest_rps * 30))
    mb_per_cp = max(0.5, rows_per_cp / 5000)  # ~5k telemetry rows per MB
    parts = {"hour": 24, "day": 1, "device_bucket": 16}[layout.partition_granularity]
    for cp in range(checkpoints):
        table.ingest_micro_batch(mb=mb_per_cp, rows=rows_per_cp,
                                 partition=f"p{cp % parts}")
    return table


def generate_dataset(n_workloads: int = 60, seed: int = 1,
                     label_noise_pct: float = 0.15):
    """Generate (layout, workload) -> measured-outcome training tuples.

    label_noise_pct adds Gaussian measurement noise proportional to each
    target's standard deviation. Production latency measurements are noisy
    (query-engine JIT, GC pauses, S3 tail latency); training without any noise
    produces a simulator R² of exactly 1.0 — the model is inverting the scan
    formula, not learning a generalizable function. A non-zero default produces
    a more honest held-out R² and tests the model's genuine signal recovery.
    """
    rng = random.Random(seed)
    noise_rng = np.random.RandomState(seed + 1000)
    layouts = candidate_layouts()
    X, y_lat, y_cost, y_wamp = [], [], [], []
    for _ in range(n_workloads):
        ingest_rps = rng.uniform(2_000, 12_000)
        avg_sel = rng.uniform(0.02, 0.6)
        tr_ratio = rng.uniform(0.3, 1.0)
        rw_ratio = rng.uniform(0.5, 8.0)
        for layout in layouts:
            table = _build_to_steady_state(layout, ingest_rps, rng)
            st = table.stats()
            wf = WorkloadFeatures(
                ingest_rows_per_sec=ingest_rps, avg_selectivity=avg_sel,
                time_range_query_ratio=tr_ratio, read_write_ratio=rw_ratio,
                avg_file_mb=st.avg_file_mb, file_count=st.file_count,
                small_file_ratio=st.small_file_ratio,
                partition_count=st.partition_count)
            X.append(encode(layout, wf))
            y_lat.append(measure_p95_latency(table))
            y_cost.append(table.storage_cost_units())
            y_wamp.append(table.bytes_rewritten_mb)

    y_lat_arr = np.array(y_lat)
    y_cost_arr = np.array(y_cost)
    y_wamp_arr = np.array(y_wamp)

    if label_noise_pct > 0.0:
        # Add proportional Gaussian noise to each target independently.
        # clip at 0 so no negative latencies or costs.
        for arr in [y_lat_arr, y_cost_arr]:
            arr += noise_rng.randn(len(arr)) * arr.std() * label_noise_pct
            np.clip(arr, 0.0, None, out=arr)
        y_wamp_arr += noise_rng.randn(len(y_wamp_arr)) * y_wamp_arr.std() * label_noise_pct
        np.clip(y_wamp_arr, 0.0, None, out=y_wamp_arr)

    return (np.array(X), y_lat_arr, y_cost_arr, y_wamp_arr)


def train(out_path: str | Path, n_workloads: int = 60, seed: int = 1) -> float:
    X, y_lat, y_cost, y_wamp = generate_dataset(n_workloads, seed)
    # held-out split for an honest R^2 gate
    n = len(X)
    idx = np.random.RandomState(seed).permutation(n)
    cut = int(n * 0.8)
    tr, te = idx[:cut], idx[cut:]
    model = CostModel().fit(X[tr], y_lat[tr], y_cost[tr], y_wamp[tr])
    r2 = model.score(X[te], y_lat[te])
    if r2 < 0.7:
        raise RuntimeError(f"cost model latency R^2={r2:.3f} below gate 0.70")
    model.save(out_path)
    return r2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/cost_model.joblib")
    ap.add_argument("--workloads", type=int, default=60)
    args = ap.parse_args()
    r2 = train(args.out, args.workloads)
    print(f"trained cost model -> {args.out}  (held-out latency R^2={r2:.3f})")


if __name__ == "__main__":
    main()
