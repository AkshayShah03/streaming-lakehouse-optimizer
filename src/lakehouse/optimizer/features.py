"""Extract the workload signals that drive layout decisions.

In production these come from: the query engine's query log (Trino
`system.runtime.queries`, Spark history server), the streaming job's
checkpoint metrics, and Iceberg `files`/`snapshots` metadata tables.
Here we model the same fields so the optimizer is trained and evaluated on
the identical feature contract it will see live.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Sequence


@dataclass(frozen=True)
class QueryLogEntry:
    selectivity: float          # fraction of bytes the query reads
    is_time_range: bool         # benefits from time partitioning
    touched_partitions: int


@dataclass(frozen=True)
class WorkloadFeatures:
    """The feature contract shared by training and serving."""
    ingest_rows_per_sec: float
    avg_selectivity: float
    time_range_query_ratio: float
    read_write_ratio: float
    avg_file_mb: float
    file_count: int
    small_file_ratio: float
    partition_count: int

    def to_vector(self) -> list[float]:
        # Stable, documented ordering -- the model depends on it.
        return [
            self.ingest_rows_per_sec,
            self.avg_selectivity,
            self.time_range_query_ratio,
            self.read_write_ratio,
            self.avg_file_mb,
            float(self.file_count),
            self.small_file_ratio,
            float(self.partition_count),
        ]

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "ingest_rows_per_sec", "avg_selectivity", "time_range_query_ratio",
            "read_write_ratio", "avg_file_mb", "file_count",
            "small_file_ratio", "partition_count",
        ]

    def as_dict(self) -> dict:
        return asdict(self)


def extract_features(
    *,
    query_log: Sequence[QueryLogEntry],
    ingest_rows_per_sec: float,
    writes_per_window: int,
    avg_file_mb: float,
    file_count: int,
    small_file_ratio: float,
    partition_count: int,
) -> WorkloadFeatures:
    if not query_log:
        avg_sel, tr_ratio = 0.5, 0.0
    else:
        avg_sel = mean(q.selectivity for q in query_log)
        tr_ratio = sum(1 for q in query_log if q.is_time_range) / len(query_log)
    reads = len(query_log)
    rw_ratio = reads / writes_per_window if writes_per_window else float(reads)
    return WorkloadFeatures(
        ingest_rows_per_sec=ingest_rows_per_sec,
        avg_selectivity=avg_sel,
        time_range_query_ratio=tr_ratio,
        read_write_ratio=rw_ratio,
        avg_file_mb=avg_file_mb,
        file_count=file_count,
        small_file_ratio=small_file_ratio,
        partition_count=partition_count,
    )
