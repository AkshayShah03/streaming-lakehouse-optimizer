"""Safe schema evolution.

Iceberg supports schema evolution by field-id, but not every change is safe
for live readers. This module encodes the subset that can be applied with
zero downtime and rejects the rest, so a producer change can be gated in CI
before it ever reaches the table -- the schema-side complement to data
contracts.

Allowed (backward compatible, metadata-only):
  * add a new OPTIONAL (nullable) column
  * widen a type along Iceberg's promotion rules (int->long, float->double,
    decimal precision increase)
  * make a required column optional

Rejected (breaks readers or loses data):
  * drop / rename a column without an explicit mapping
  * narrow a type (long->int)
  * make an optional column required
  * change a field id
"""
from __future__ import annotations

from dataclasses import dataclass


_WIDENING = {
    ("int", "long"), ("float", "double"),
    ("decimal", "decimal"),  # only if precision increases (checked below)
}


@dataclass(frozen=True)
class Field:
    field_id: int
    name: str
    type: str
    required: bool
    precision: int = 0  # for decimal


class SchemaEvolutionError(Exception):
    pass


def validate_evolution(old: list[Field], new: list[Field]) -> None:
    """Raise SchemaEvolutionError if `new` is not a backward-compatible
    evolution of `old`. No-op (returns None) if safe."""
    old_by_id = {f.field_id: f for f in old}
    new_by_id = {f.field_id: f for f in new}

    # Dropped columns
    dropped = set(old_by_id) - set(new_by_id)
    if dropped:
        names = ", ".join(old_by_id[i].name for i in dropped)
        raise SchemaEvolutionError(f"column drop not allowed without mapping: {names}")

    for fid, of in old_by_id.items():
        nf = new_by_id[fid]
        # Rename via field-id is allowed by Iceberg, but flag silent renames
        # so they are reviewed, not slipped through.
        if of.name != nf.name:
            raise SchemaEvolutionError(
                f"field {fid} renamed {of.name!r}->{nf.name!r}; requires explicit review")
        # Type changes
        if of.type != nf.type:
            if (of.type, nf.type) not in _WIDENING:
                raise SchemaEvolutionError(
                    f"unsafe type change on {of.name}: {of.type}->{nf.type}")
        if of.type == "decimal" and nf.type == "decimal" and nf.precision < of.precision:
            raise SchemaEvolutionError(
                f"decimal precision narrowed on {of.name}: {of.precision}->{nf.precision}")
        # Nullability: required->optional ok; optional->required is unsafe
        if not of.required and nf.required:
            raise SchemaEvolutionError(
                f"column {of.name} made required; existing nulls would break")

    # New columns must be optional
    added = set(new_by_id) - set(old_by_id)
    for fid in added:
        if new_by_id[fid].required:
            raise SchemaEvolutionError(
                f"new column {new_by_id[fid].name} must be optional/nullable")


def is_compatible(old: list[Field], new: list[Field]) -> bool:
    try:
        validate_evolution(old, new)
        return True
    except SchemaEvolutionError:
        return False
