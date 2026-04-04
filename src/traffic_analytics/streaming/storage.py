from __future__ import annotations

from uuid import UUID

import pandas as pd
from sqlalchemy import text

from traffic_analytics.config.settings import Settings
from traffic_analytics.streaming.metrics import build_streaming_metrics
from traffic_analytics.streaming.preannotated_backend import build_tracks


EVENT_COLUMNS = [
    "run_id",
    "video_source_id",
    "camera_id",
    "frame_ts",
    "frame_no",
    "track_id",
    "source_track_id",
    "class_name_raw",
    "vehicle_class",
    "confidence",
    "centroid_x",
    "centroid_y",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "direction_hint",
    "speed_proxy",
    "event_source",
]

TRACK_COLUMNS = [
    "run_id",
    "video_source_id",
    "camera_id",
    "track_id",
    "source_track_id",
    "vehicle_class",
    "first_seen_ts",
    "last_seen_ts",
    "duration_seconds",
    "detections_count",
    "start_centroid_x",
    "start_centroid_y",
    "end_centroid_x",
    "end_centroid_y",
    "direction",
    "distance_proxy",
    "speed_proxy",
    "speed_kmh_estimated",
    "event_source",
]

METRIC_COLUMNS = [
    "run_id",
    "camera_id",
    "window_start",
    "window_end",
    "window_granularity",
    "total_count",
    "car_count",
    "truck_count",
    "bus_count",
    "motorcycle_count",
    "bicycle_count",
    "person_count",
    "avg_speed_proxy",
    "avg_speed_kmh",
    "movement_score_avg",
    "occupancy_proxy",
    "congestion_proxy",
    "heavy_vehicle_share",
]


def prepare_events(
    detected_events: pd.DataFrame,
    run_id: UUID,
    video_source_id: int,
    camera_id: str,
    event_source: str,
    track_prefix: str = "",
) -> pd.DataFrame:
    events = detected_events.copy()
    if track_prefix:
        events["track_id"] = events["track_id"].map(lambda value: f"{track_prefix}{value}")
    events["run_id"] = str(run_id)
    events["video_source_id"] = video_source_id
    events["camera_id"] = camera_id
    events["source_track_id"] = None
    events["direction_hint"] = None
    events["event_source"] = event_source
    return events[EVENT_COLUMNS]


def build_tracks_from_events(
    events: pd.DataFrame,
    run_id: UUID,
    video_source_id: int,
    camera_id: str,
    event_source: str,
) -> pd.DataFrame:
    tracks_source = events.rename(columns={"frame_ts": "detection_time"})
    tracks_source["x_cord_m"] = tracks_source["centroid_x"]
    tracks_source["y_cord_m"] = tracks_source["centroid_y"]
    tracks_source["direction"] = None
    tracks = build_tracks(
        tracks_source[
            ["track_id", "vehicle_class", "detection_time", "x_cord_m", "y_cord_m", "direction"]
        ]
    )
    if tracks.empty:
        return tracks
    tracks["run_id"] = str(run_id)
    tracks["video_source_id"] = video_source_id
    tracks["camera_id"] = camera_id
    tracks["event_source"] = event_source
    return tracks[TRACK_COLUMNS]


def summarize_detected_tracks(detected_events: pd.DataFrame) -> pd.DataFrame:
    if detected_events.empty:
        return pd.DataFrame()
    tracks_source = detected_events.rename(columns={"frame_ts": "detection_time"})
    tracks_source["x_cord_m"] = tracks_source["centroid_x"]
    tracks_source["y_cord_m"] = tracks_source["centroid_y"]
    tracks_source["direction"] = None
    return build_tracks(
        tracks_source[
            ["track_id", "vehicle_class", "detection_time", "x_cord_m", "y_cord_m", "direction"]
        ]
    )


def append_detection_events(engine, events: pd.DataFrame) -> int:
    if events.empty:
        return 0
    events.to_sql("detection_events", engine, schema="core", if_exists="append", index=False, method="multi")
    return len(events)


def append_tracks(engine, tracks: pd.DataFrame) -> int:
    if tracks.empty:
        return 0
    tracks.to_sql("tracked_objects", engine, schema="core", if_exists="append", index=False, method="multi")
    return len(tracks)


def replace_run_metrics(engine, run_id: UUID, tracks_for_run: pd.DataFrame, settings: Settings) -> int:
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM dm.streaming_metrics WHERE run_id = :run_id"), {"run_id": str(run_id)})
    if tracks_for_run.empty:
        return 0
    metrics = build_streaming_metrics(tracks_for_run, settings.stream_camera_id, settings.stream_window_minutes)
    if metrics.empty:
        return 0
    metrics["run_id"] = str(run_id)
    metrics = metrics[METRIC_COLUMNS]
    metrics.to_sql("streaming_metrics", engine, schema="dm", if_exists="append", index=False, method="multi")
    return len(metrics)
