CREATE TABLE IF NOT EXISTS dm.streaming_metrics (
    metric_id BIGSERIAL PRIMARY KEY,
    run_id UUID REFERENCES ops.pipeline_runs(run_id),
    camera_id TEXT NOT NULL,
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    window_granularity TEXT NOT NULL,
    total_count INTEGER NOT NULL,
    car_count INTEGER NOT NULL DEFAULT 0,
    truck_count INTEGER NOT NULL DEFAULT 0,
    bus_count INTEGER NOT NULL DEFAULT 0,
    motorcycle_count INTEGER NOT NULL DEFAULT 0,
    bicycle_count INTEGER NOT NULL DEFAULT 0,
    person_count INTEGER NOT NULL DEFAULT 0,
    avg_speed_proxy NUMERIC,
    avg_speed_kmh NUMERIC,
    movement_score_avg NUMERIC,
    occupancy_proxy NUMERIC,
    congestion_proxy NUMERIC,
    heavy_vehicle_share NUMERIC,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_streaming_metrics_window
    ON dm.streaming_metrics (camera_id, window_start);

CREATE TABLE IF NOT EXISTS dm.batch_metrics (
    metric_id BIGSERIAL PRIMARY KEY,
    run_id UUID REFERENCES ops.pipeline_runs(run_id),
    camera_id TEXT NOT NULL,
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    window_granularity TEXT NOT NULL,
    total_count INTEGER NOT NULL,
    car_count INTEGER NOT NULL DEFAULT 0,
    truck_count INTEGER NOT NULL DEFAULT 0,
    bus_count INTEGER NOT NULL DEFAULT 0,
    motorcycle_count INTEGER NOT NULL DEFAULT 0,
    bicycle_count INTEGER NOT NULL DEFAULT 0,
    person_count INTEGER NOT NULL DEFAULT 0,
    avg_speed_proxy NUMERIC,
    avg_speed_kmh NUMERIC,
    movement_score_avg NUMERIC,
    occupancy_proxy NUMERIC,
    congestion_proxy NUMERIC,
    heavy_vehicle_share NUMERIC,
    weather_type TEXT,
    weather_code INTEGER,
    temperature NUMERIC,
    precipitation NUMERIC,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_batch_metrics_window
    ON dm.batch_metrics (camera_id, window_start);

