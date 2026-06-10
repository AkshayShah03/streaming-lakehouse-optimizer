"""CLI wrapper around the canonical benchmark suite (src/lakehouse/maintenance/benchmark.py).

Run `python -m benchmarks.query_suite` to print the workload mix the
optimizer is tuned against. The authoritative definitions live in the
package so training and shadow-eval cannot drift apart.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lakehouse.maintenance.benchmark import DEFAULT_SUITE  # noqa: E402


def main() -> None:
    print(f"{'query':<22}{'selectivity':>12}{'time_range':>12}{'weight':>8}")
    for q in DEFAULT_SUITE:
        print(f"{q.name:<22}{q.selectivity:>12}{str(q.is_time_range):>12}{q.weight:>8}")


if __name__ == "__main__":
    main()
