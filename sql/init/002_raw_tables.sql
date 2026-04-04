CREATE TABLE IF NOT EXISTS ops.pipeline_runs (
    run_id UUID PRIMARY KEY,
    pipeline_name TEXT NOT NULL,
    pipeline_mode TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ,
    source_ref TEXT,
    rows_written INTEGER DEFAULT 0,
    error_text TEXT,
    config_snapshot_json JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS ix_pipeline_runs_name_started
    ON ops.pipeline_runs (pipeline_name, started_at DESC);

CREATE INDEX IF NOT EXISTS ix_pipeline_runs_status_started
    ON ops.pipeline_runs (status, started_at DESC);

CREATE TABLE IF NOT EXISTS ops.data_quality_checks (
    check_id BIGSERIAL PRIMARY KEY,
    run_id UUID REFERENCES ops.pipeline_runs(run_id),
    dataset_name TEXT NOT NULL,
    check_name TEXT NOT NULL,
    check_scope TEXT NOT NULL,
    check_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status TEXT NOT NULL,
    actual_value NUMERIC,
    threshold_value NUMERIC,
    details_json JSONB DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS raw.video_sources (
    video_source_id BIGSERIAL PRIMARY KEY,
    source_type TEXT NOT NULL,
    requested_uri TEXT,
    resolved_uri TEXT,
    local_path TEXT,
    status TEXT NOT NULL,
    message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    duration_sec NUMERIC,
    fps NUMERIC,
    width INTEGER,
    height INTEGER
);

CREATE TABLE IF NOT EXISTS raw.batch_imports (
    import_id BIGSERIAL PRIMARY KEY,
    dataset_name TEXT NOT NULL,
    source_path TEXT NOT NULL,
    file_hash TEXT,
    loaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    row_count INTEGER,
    status TEXT NOT NULL,
    run_id UUID REFERENCES ops.pipeline_runs(run_id)
);

CREATE TABLE IF NOT EXISTS raw.tracking_source_rows (
    source_row_id BIGSERIAL PRIMARY KEY,
    run_id UUID REFERENCES ops.pipeline_runs(run_id),
    video_id BIGINT,
    title TEXT,
    path TEXT,
    recording_at TIMESTAMPTZ,
    recording_end TIMESTAMPTZ,
    track_id BIGINT,
    tracker_id BIGINT,
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    object_type TEXT,
    width NUMERIC,
    length NUMERIC,
    detection_time TIMESTAMPTZ,
    x_cord_m NUMERIC,
    y_cord_m NUMERIC,
    total_kms NUMERIC,
    speed_km_h NUMERIC,
    direction TEXT,
    source_file TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_tracking_source_rows_detection_time
    ON raw.tracking_source_rows (detection_time);

CREATE INDEX IF NOT EXISTS ix_tracking_source_rows_track_id
    ON raw.tracking_source_rows (track_id);

CREATE TABLE IF NOT EXISTS raw.traffic_30min_source (
    source_row_id BIGSERIAL PRIMARY KEY,
    run_id UUID REFERENCES ops.pipeline_runs(run_id),
    timestamp_ts TIMESTAMPTZ NOT NULL,
    date_value DATE,
    time_value TIME,
    avg_speed NUMERIC,
    weather_type TEXT,
    weather_code INTEGER,
    temperature NUMERIC,
    precipitation NUMERIC,
    intensity_30min INTEGER,
    cars INTEGER,
    trucks INTEGER,
    busses INTEGER,
    source_file TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_traffic_30min_source_timestamp
    ON raw.traffic_30min_source (timestamp_ts);

