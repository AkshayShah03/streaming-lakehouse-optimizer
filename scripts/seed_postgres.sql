-- Source table captured by Debezium. REPLICA IDENTITY FULL so updates/deletes
-- carry full before-images (needed for correct CDC downstream).
CREATE TABLE IF NOT EXISTS telemetry (
    id            BIGSERIAL PRIMARY KEY,
    vehicle_id    BIGINT      NOT NULL,
    ts_ms         BIGINT      NOT NULL,
    speed_kph     DOUBLE PRECISION,
    battery_pct   DOUBLE PRECISION,
    motor_temp_c  DOUBLE PRECISION,
    pack_voltage  DOUBLE PRECISION,
    odometer_km   DOUBLE PRECISION,
    lat           DOUBLE PRECISION,
    lon           DOUBLE PRECISION
);
ALTER TABLE telemetry REPLICA IDENTITY FULL;
CREATE INDEX IF NOT EXISTS idx_telemetry_ts ON telemetry (ts_ms);
