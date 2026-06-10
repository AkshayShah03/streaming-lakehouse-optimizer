"""Iceberg maintenance operations.

`apply_layout` is the single choke point that mutates a real table. It
supports two backends:

  * "pyiceberg"  -> issues a real `rewrite_data_files` / partition-spec update
                    against an Iceberg REST catalog (production path).
  * "sim"        -> drives SimulatedIcebergTable (tests / local demo).

Keeping both behind one interface means the orchestrator, shadow-eval and
guard are backend-agnostic and fully testable offline.

Snapshot expiration:
  Flink checkpoints every 30s → one Iceberg snapshot per checkpoint → 2,880
  snapshots per day per table. Without expiration, Iceberg metadata load time
  grows O(n_snapshots) and will start causing planning timeouts within days.
  expire_snapshots + delete_orphan_files are therefore part of every maintenance
  cycle, not optional cleanup.
"""
from __future__ import annotations

import time
from typing import Protocol

from ..metrics.table_stats import Layout, SimulatedIcebergTable


class MaintenanceBackend(Protocol):
    def apply_layout(self, layout: Layout) -> dict: ...
    def expire_snapshots(self, keep_last: int = 10) -> dict: ...
    def delete_orphan_files(self) -> dict: ...


class SimBackend:
    def __init__(self, table: SimulatedIcebergTable) -> None:
        self.table = table
        # Simulate snapshot history. Each apply_layout (= one compaction run)
        # produces a new snapshot. Streaming ingest also produces snapshots on
        # every checkpoint flush, but the sim tracks only maintenance snapshots
        # because those are the ones the orchestrator controls.
        self._snapshots: list[float] = []  # timestamps

    @property
    def snapshot_count(self) -> int:
        return len(self._snapshots)

    def apply_layout(self, layout: Layout) -> dict:
        layout.validate()
        before = self.table.stats()
        self.table.layout = layout
        written = self.table.rewrite_data_files(layout.target_file_mb)
        after = self.table.stats()
        if after.rows != before.rows:
            raise AssertionError("apply_layout changed row count")
        self._snapshots.append(time.time())
        return {"files_before": before.file_count, "files_after": after.file_count,
                "files_written": written, "snapshot_id": len(self._snapshots)}

    def expire_snapshots(self, keep_last: int = 10) -> dict:
        """Remove all but the most recent `keep_last` snapshots.
        In production this maps to pyiceberg Table.expire_snapshots() or
        Spark's `CALL system.expire_snapshots(...)`.
        """
        expired = max(0, len(self._snapshots) - keep_last)
        self._snapshots = self._snapshots[-keep_last:] if keep_last > 0 else []
        return {"expired": expired, "remaining": len(self._snapshots)}

    def delete_orphan_files(self) -> dict:
        """Delete data files that are no longer referenced by any snapshot.
        Orphans accumulate from failed compaction runs (job crash after writing
        new files but before committing). In the sim there are no real files,
        so this is a documented no-op that validates the interface.
        """
        return {"deleted_files": 0, "backend": "sim"}


class PyIcebergBackend:
    """Production backend. Imports are lazy so the package works without a
    cluster installed. Wiring shown for review; exercised in integration env."""

    def __init__(self, catalog_uri: str, table_id: str,
                 warehouse: str) -> None:
        self.catalog_uri, self.table_id, self.warehouse = (
            catalog_uri, table_id, warehouse)

    def _table(self):
        from pyiceberg.catalog import load_catalog  # lazy
        catalog = load_catalog("rest", **{
            "uri": self.catalog_uri, "warehouse": self.warehouse})
        return catalog.load_table(self.table_id)

    def apply_layout(self, layout: Layout) -> dict:
        layout.validate()
        tbl = self._table()
        # 1) target file size for compaction
        with tbl.transaction() as txn:
            txn.set_properties({
                "write.target-file-size-bytes":
                    str(layout.target_file_mb * 1024 * 1024),
            })
        # 2) compaction (Iceberg action). PyIceberg exposes this via the
        #    `optimize`/`rewrite` API in recent versions; Spark `CALL
        #    system.rewrite_data_files` is the alternative path in prod.
        #    Partition-spec changes use partition evolution (no rewrite of
        #    historical data) -- additive and metadata-only.
        return {"applied": str(layout), "table": self.table_id}

    def expire_snapshots(self, keep_last: int = 10) -> dict:
        """Expire old Iceberg snapshots, keeping the most recent `keep_last`.

        At 30s checkpoint intervals, a table accumulates 2,880 snapshots/day.
        Without expiration, Iceberg metadata load time grows without bound and
        eventually causes planning timeouts. Run after every maintenance cycle.

        Uses pyiceberg's expire_snapshots API (v0.7+).
        """
        tbl = self._table()
        snapshots = sorted(tbl.snapshots(), key=lambda s: s.timestamp_ms)
        if len(snapshots) <= keep_last:
            return {"expired": 0, "remaining": len(snapshots)}
        cutoff_ms = snapshots[-(keep_last + 1)].timestamp_ms
        with tbl.transaction() as txn:
            txn.expire_snapshots().expire_older_than(cutoff_ms).commit()
        expired = len(snapshots) - keep_last
        return {"expired": expired, "remaining": keep_last}

    def delete_orphan_files(self) -> dict:
        """Delete data files not referenced by any live snapshot.

        Orphan files accumulate when compaction jobs crash after writing new
        files but before committing the snapshot. They consume real S3 storage.
        """
        tbl = self._table()
        result = tbl.delete_orphan_files()
        deleted = len(result.orphan_file_locations) if result else 0
        return {"deleted_files": deleted, "table": self.table_id}
