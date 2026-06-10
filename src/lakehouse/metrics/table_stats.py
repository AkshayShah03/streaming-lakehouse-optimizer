"""Iceberg table statistics + a deterministic simulator.

In production, `TableStats` is hydrated from Iceberg metadata
(`table.current_snapshot()`, `files` metadata table). For local runs and
CI, `SimulatedIcebergTable` reproduces the physically relevant behaviour of
small-file accumulation, compaction (`rewrite_data_files`) and scan cost so
the optimizer + guardrail loop can be exercised end-to-end with no cluster.

The simulator is intentionally a *physical model*, not random noise: the
learned cost model has a real signal to recover, which is what makes the
unit tests meaningful rather than tautological.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import Optional


# ---- Layout knobs the optimizer is allowed to choose ----------------------

@dataclass(frozen=True)
class Layout:
    """A physical layout decision for one table."""
    target_file_mb: int          # compaction target file size
    compaction_trigger_files: int  # rewrite once this many small files accrue
    partition_granularity: str   # 'hour' | 'day' | 'device_bucket'

    def validate(self) -> None:
        if not (16 <= self.target_file_mb <= 1024):
            raise ValueError(f"target_file_mb out of range: {self.target_file_mb}")
        if self.compaction_trigger_files < 2:
            raise ValueError("compaction_trigger_files must be >= 2")
        if self.partition_granularity not in {"hour", "day", "device_bucket"}:
            raise ValueError(f"bad partition granularity: {self.partition_granularity}")


DEFAULT_LAYOUT = Layout(target_file_mb=128, compaction_trigger_files=50,
                        partition_granularity="day")


@dataclass(frozen=True)
class TableStats:
    """Snapshot of physical table state (mirrors Iceberg `files` metadata)."""
    file_count: int
    total_mb: float
    partition_count: int
    rows: int

    @property
    def avg_file_mb(self) -> float:
        return self.total_mb / self.file_count if self.file_count else 0.0

    @property
    def small_file_ratio(self) -> float:
        # Fraction of files well below the canonical 128MB target.
        return _small_ratio(self.avg_file_mb)


def _small_ratio(avg_file_mb: float) -> float:
    if avg_file_mb >= 128:
        return 0.0
    return max(0.0, min(1.0, (128 - avg_file_mb) / 128))


# ---- The simulator --------------------------------------------------------

@dataclass
class _File:
    mb: float
    rows: int
    partition: str


class SimulatedIcebergTable:
    """A toy Iceberg table that accumulates small files on ingest and can be
    compacted. Scan cost reflects per-file open overhead + bytes scanned,
    which is exactly why the small-file problem hurts."""

    # Tunable physical constants (ms). These are the "ground truth" the cost
    # model must learn to approximate from observed signals.
    FILE_OPEN_MS = 4.0          # planning + open cost paid per file touched
    MB_SCAN_MS = 0.35           # cost per MB actually read
    STORAGE_COST_PER_GB = 0.023  # $/GB-month (S3-ish)

    def __init__(self, layout: Layout = DEFAULT_LAYOUT, seed: int = 7) -> None:
        layout.validate()
        self.layout = layout
        self._files: list[_File] = []
        self._rng = random.Random(seed)
        # write amplification accumulator (bytes rewritten by compaction)
        self.bytes_rewritten_mb: float = 0.0

    # -- ingest ------------------------------------------------------------
    def ingest_micro_batch(self, mb: float, rows: int, partition: str) -> None:
        """A streaming checkpoint flush => one small file (the root cause)."""
        self._files.append(_File(mb=mb, rows=rows, partition=partition))
        self._maybe_autocompact()

    def _maybe_autocompact(self) -> None:
        if len(self._files) >= self.layout.compaction_trigger_files:
            self.rewrite_data_files(self.layout.target_file_mb)

    # -- maintenance -------------------------------------------------------
    def rewrite_data_files(self, target_file_mb: int) -> int:
        """Bin-pack small files into target-sized files. Returns #files written.
        Mirrors Iceberg's `rewriteDataFiles` action and preserves all rows."""
        if not self._files:
            return 0
        by_part: dict[str, list[_File]] = {}
        for f in self._files:
            by_part.setdefault(f.partition, []).append(f)

        new_files: list[_File] = []
        for part, files in by_part.items():
            total_mb = sum(f.mb for f in files)
            total_rows = sum(f.rows for f in files)
            self.bytes_rewritten_mb += total_mb  # write amplification
            n = max(1, math.ceil(total_mb / target_file_mb))
            for i in range(n):
                share = (i + 1) / n
                prev = i / n
                new_files.append(_File(
                    mb=total_mb * (share - prev),
                    rows=round(total_rows * (share - prev)),
                    partition=part,
                ))
            # fix rounding so row count is exactly preserved (data-loss guard)
            delta = total_rows - sum(nf.rows for nf in new_files if nf.partition == part)
            if delta and new_files:
                same = [nf for nf in new_files if nf.partition == part]
                same[-1].rows += delta
        self._files = new_files
        return len(new_files)

    # -- read --------------------------------------------------------------
    def scan(self, selectivity: float, partition_pruned: bool = True) -> float:
        """Return simulated scan latency (ms) for a query that reads
        `selectivity` fraction of bytes. Per-file overhead dominates when
        files are tiny -- the effect the optimizer exploits."""
        files = self._files
        if partition_pruned and self.layout.partition_granularity != "device_bucket":
            # finer partitioning prunes more files for time-range queries
            prune = 0.6 if self.layout.partition_granularity == "hour" else 0.3
            files = self._files[: max(1, int(len(self._files) * (1 - prune)))]
        mb_read = sum(f.mb for f in files) * selectivity
        return len(files) * self.FILE_OPEN_MS + mb_read * self.MB_SCAN_MS

    # -- introspection -----------------------------------------------------
    def stats(self) -> TableStats:
        return TableStats(
            file_count=len(self._files),
            total_mb=sum(f.mb for f in self._files),
            partition_count=len({f.partition for f in self._files}),
            rows=sum(f.rows for f in self._files),
        )

    def storage_cost_units(self) -> float:
        """Monthly $ for steady-state LIVE storage only.

        Write-amplification (the one-time cost of over-aggressive compaction)
        is deliberately kept OUT of this number and tracked as its own guarded
        metric (`bytes_rewritten_mb`). Conflating them double-penalizes a
        beneficial first compaction and lets the guard reject genuine wins."""
        live_gb = self.stats().total_mb / 1024
        return live_gb * self.STORAGE_COST_PER_GB

    def clone(self) -> "SimulatedIcebergTable":
        """Deep copy for shadow evaluation (never mutate production)."""
        c = SimulatedIcebergTable(layout=self.layout)
        c._files = [replace(f) for f in self._files]
        c.bytes_rewritten_mb = self.bytes_rewritten_mb
        return c
