"""End-to-end local demo with NO cluster required.

  1. Train the cost model on the physical simulator.
  2. Stream telemetry micro-batches into a simulated Iceberg table under a
     deliberately bad incumbent layout (tiny files, coarse partitioning).
  3. Run the guarded maintenance loop once and show the decision + ledger.

Run:  python scripts/run_local_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lakehouse.ingest.telemetry_generator import generate
from lakehouse.maintenance.benchmark import measure_p95_latency
from lakehouse.maintenance.iceberg_ops import SimBackend
from lakehouse.maintenance.orchestrator import run_once
from lakehouse.maintenance.regression_guard import GuardConfig
from lakehouse.metrics.table_stats import Layout, SimulatedIcebergTable
from lakehouse.optimizer.cost_model import CostModel
from lakehouse.optimizer.features import WorkloadFeatures
from lakehouse.optimizer.train import train


def main() -> None:
    model_path = "artifacts/cost_model.joblib"
    print("[1/3] training cost model on physical simulator ...")
    r2 = train(model_path, n_workloads=40)
    print(f"      held-out latency R^2 = {r2:.3f}")
    model = CostModel.load(model_path)

    print("[2/3] streaming telemetry into a badly-laid-out table ...")
    bad = Layout(target_file_mb=64, compaction_trigger_files=100,
                 partition_granularity="device_bucket")
    table = SimulatedIcebergTable(layout=bad)
    rows = 0
    for batch in generate(rps=10_000, seconds=30, fleet_size=2000):
        mb = len(batch) / 5000
        table.ingest_micro_batch(mb=mb, rows=len(batch),
                                 partition=f"bucket{batch[0].vehicle_id % 16}")
        rows += len(batch)
    st = table.stats()
    print(f"      ingested {rows} rows -> {st.file_count} files, "
          f"avg {st.avg_file_mb:.1f}MB, p95 {measure_p95_latency(table):.1f}ms")

    print("[3/3] running guarded optimize loop ...")
    wf = WorkloadFeatures(
        ingest_rows_per_sec=10_000, avg_selectivity=0.08,
        time_range_query_ratio=0.9, read_write_ratio=4.0,
        avg_file_mb=st.avg_file_mb, file_count=st.file_count,
        small_file_ratio=st.small_file_ratio, partition_count=st.partition_count)

    result = run_once(model=model, production=table, backend=SimBackend(table),
                      wf=wf, guard=GuardConfig(min_improvement_pct=5.0))
    print(f"      proposed : {result.proposed}")
    print(f"      promoted : {result.promoted}")
    print(f"      reason   : {result.reason}")
    if result.promoted:
        print(f"      after    : p95 {measure_p95_latency(table):.1f}ms, "
              f"{table.stats().file_count} files")
    print("      ledger   : artifacts/promotion_ledger.jsonl")


if __name__ == "__main__":
    main()
