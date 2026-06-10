"""Exactly-once sink semantics, modeled and unit-testable.

Flink's Kafka source + Iceberg sink give end-to-end exactly-once via
checkpoint-aligned, two-phase commits: each committed data file is tagged
with the checkpoint id, and on recovery uncommitted files are discarded
while already-committed checkpoints are never re-applied.

We model the *commit-idempotency* invariant that makes this correct: a
replay of a checkpoint that was already committed is a no-op, and a partial
write that never committed contributes no rows. The real sink delegates to
Flink; this class lets us assert the invariant in CI and document exactly
what "exactly-once" means here.

Abort semantics (fix for phantom-checkpoint bug):
  abort(id) marks the checkpoint as aborted; a subsequent commit(id) without
  an intervening re-stage is a safe no-op (returns False, no committed marker).
  This prevents the phantom case where abort + commit left id in
  _committed_checkpoints, blocking a legitimate recovery re-play of the same id.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CommittedFile:
    checkpoint_id: int
    rows: int


class ExactlyOnceSink:
    def __init__(self) -> None:
        self._committed_checkpoints: set[int] = set()
        self._aborted_checkpoints: set[int] = set()
        self._files: list[CommittedFile] = []
        self._staged: dict[int, int] = {}  # checkpoint_id -> staged rows

    def stage(self, checkpoint_id: int, rows: int) -> None:
        """Pre-commit phase: buffer rows for a checkpoint (not yet visible).
        Re-staging an aborted checkpoint clears its aborted status so recovery
        can commit it cleanly."""
        self._aborted_checkpoints.discard(checkpoint_id)
        self._staged[checkpoint_id] = self._staged.get(checkpoint_id, 0) + rows

    def commit(self, checkpoint_id: int) -> bool:
        """Two-phase commit. Idempotent: committing an already-committed
        checkpoint is a no-op. Returns True iff this commit changed state.

        Committing an aborted-but-not-restaged checkpoint is also a no-op
        (returns False) so the id is NOT added to _committed_checkpoints and
        a future recovery re-stage+commit can succeed."""
        if checkpoint_id in self._committed_checkpoints:
            self._staged.pop(checkpoint_id, None)
            return False  # replay after recovery -> no duplicate

        if checkpoint_id in self._aborted_checkpoints and \
                checkpoint_id not in self._staged:
            # abort without re-stage: safe no-op, no phantom committed marker.
            return False

        rows = self._staged.pop(checkpoint_id, 0)
        self._files.append(CommittedFile(checkpoint_id, rows))
        self._committed_checkpoints.add(checkpoint_id)
        self._aborted_checkpoints.discard(checkpoint_id)
        return True

    def abort(self, checkpoint_id: int) -> None:
        """Failure before commit: staged rows are discarded (no partial write).
        Marks the id as aborted so commit(id) without re-stage is a no-op."""
        self._staged.pop(checkpoint_id, None)
        self._aborted_checkpoints.add(checkpoint_id)

    @property
    def total_rows(self) -> int:
        return sum(f.rows for f in self._files)

    @property
    def committed_checkpoints(self) -> int:
        return len(self._committed_checkpoints)
