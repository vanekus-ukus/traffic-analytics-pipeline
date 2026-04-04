CREATE TABLE IF NOT EXISTS core.detection_events (
    event_id BIGSERIAL PRIMARY KEY,
    run_id UUID REFERENCES ops.pipeline_runs(run_id),
    video_source_id BIGINT REFERENCES raw.video_sources(video_source_id),
    camera_id TEXT NOT NULL,
    frame_ts TIMESTAMPTZ NOT NULL,
    frame_no INTEGER,
    track_id TEXT,
    source_track_id BIGINT,
    class_name_raw TEXT,
    vehicle_class TEXT NOT NULL,
    confidence NUMERIC,
    centroid_x NUMERIC,
    centroid_y NUMERIC,
    bbox_x1 NUMERIC,
    bbox_y1 NUMERIC,
    bbox_x2 NUMERIC,
    bbox_y2 NUMERIC,
    direction_hint TEXT,
    speed_proxy NUMERIC,
    event_source TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_detection_events_camera_ts
    ON core.detection_events (camera_id, frame_ts);

CREATE INDEX IF NOT EXISTS ix_detection_events_track_ts
    ON core.detection_events (track_id, frame_ts);

CREATE TABLE IF NOT EXISTS core.tracked_objects (
    tracked_object_id BIGSERIAL PRIMARY KEY,
    run_id UUID REFERENCES ops.pipeline_runs(run_id),
    video_source_id BIGINT REFERENCES raw.video_sources(video_source_id),
    camera_id TEXT NOT NULL,
    track_id TEXT NOT NULL,
    source_track_id BIGINT,
    vehicle_class TEXT NOT NULL,
    first_seen_ts TIMESTAMPTZ NOT NULL,
    last_seen_ts TIMESTAMPTZ NOT NULL,
    duration_seconds NUMERIC,
    detections_count INTEGER NOT NULL,
    start_centroid_x NUMERIC,
    start_centroid_y NUMERIC,
    end_centroid_x NUMERIC,
    end_centroid_y NUMERIC,
    direction TEXT,
    distance_proxy NUMERIC,
    speed_proxy NUMERIC,
    speed_kmh_estimated NUMERIC,
    event_source TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_tracked_objects_camera_first_seen
    ON core.tracked_objects (camera_id, first_seen_ts);

CREATE INDEX IF NOT EXISTS ix_tracked_objects_vehicle_first_seen
    ON core.tracked_objects (vehicle_class, first_seen_ts);

