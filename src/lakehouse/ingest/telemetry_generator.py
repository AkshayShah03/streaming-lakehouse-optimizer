"""Vehicle telemetry generator -- the CDC source.

Writes realistic fleet telemetry into Postgres; Debezium captures the WAL
and streams change events to Kafka. Designed to sustain 10k+ events/sec on
a laptop by batching inserts. Schema mirrors `scripts/seed_postgres.sql`.

Run:  python -m lakehouse.ingest.telemetry_generator --rps 10000 --seconds 60
"""
from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass


@dataclass
class TelemetryRow:
    vehicle_id: int
    ts_ms: int
    speed_kph: float
    battery_pct: float
    motor_temp_c: float
    pack_voltage: float
    odometer_km: float
    lat: float
    lon: float


def _row(vehicle_id: int, t: float, rng: random.Random) -> TelemetryRow:
    # smooth-ish signals so downstream aggregates are non-degenerate
    speed = max(0.0, 60 + 40 * math.sin(t / 30 + vehicle_id) + rng.gauss(0, 5))
    return TelemetryRow(
        vehicle_id=vehicle_id,
        ts_ms=int(t * 1000),
        speed_kph=round(speed, 2),
        battery_pct=round(max(0, 80 - (t / 3600) * 12 + rng.gauss(0, 0.5)), 2),
        motor_temp_c=round(35 + speed * 0.25 + rng.gauss(0, 2), 2),
        pack_voltage=round(355 + rng.gauss(0, 1.5), 2),
        odometer_km=round(10_000 + speed * t / 3600, 2),
        lat=round(30.0 + rng.gauss(0, 0.05), 6),   # Houston-ish, fittingly
        lon=round(-95.4 + rng.gauss(0, 0.05), 6),
    )


def generate(rps: int, seconds: int, fleet_size: int = 5000, seed: int = 0):
    """Yield batches of rows at the requested rate. Pure generator -- the
    DB-writing wrapper is in `main` so the rate logic is unit-testable."""
    rng = random.Random(seed)
    batch_per_sec = 10
    per_batch = max(1, rps // batch_per_sec)
    start = time.time()
    for tick in range(seconds * batch_per_sec):
        t = tick / batch_per_sec
        batch = [_row(rng.randint(0, fleet_size - 1), t, rng)
                 for _ in range(per_batch)]
        yield batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rps", type=int, default=10_000)
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--dsn", default="postgresql://lake:lake@localhost:5432/fleet")
    args = ap.parse_args()
    try:
        import psycopg2
        from psycopg2.extras import execute_values
    except ImportError:
        raise SystemExit("pip install psycopg2-binary to run the live generator")

    conn = psycopg2.connect(args.dsn)
    conn.autocommit = True
    cur = conn.cursor()
    sql = ("INSERT INTO telemetry (vehicle_id, ts_ms, speed_kph, battery_pct, "
           "motor_temp_c, pack_voltage, odometer_km, lat, lon) VALUES %s")
    sent = 0
    t0 = time.time()
    for batch in generate(args.rps, args.seconds):
        execute_values(cur, sql, [(
            r.vehicle_id, r.ts_ms, r.speed_kph, r.battery_pct, r.motor_temp_c,
            r.pack_voltage, r.odometer_km, r.lat, r.lon) for r in batch])
        sent += len(batch)
        time.sleep(max(0, (sent / args.rps) - (time.time() - t0)))
    print(f"inserted {sent} rows in {time.time()-t0:.1f}s "
          f"(~{sent/(time.time()-t0):.0f} rows/s)")


if __name__ == "__main__":
    main()
