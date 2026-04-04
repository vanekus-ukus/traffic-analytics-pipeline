from __future__ import annotations

import argparse
from dataclasses import asdict
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from traffic_analytics.common.logging_utils import configure_logging
from traffic_analytics.config.settings import get_settings
from traffic_analytics.db.engine import get_engine
from traffic_analytics.db.runtime import finish_pipeline_run, record_quality_check, start_pipeline_run
from traffic_analytics.streaming.metrics import build_streaming_metrics
from traffic_analytics.streaming.preannotated_backend import build_tracks, load_preannotated_events
from traffic_analytics.streaming.video_source import register_video_source, resolve_video_source
from traffic_analytics.streaming.yolo_backend import detect_and_track_video

LOGGER = logging.getLogger(__name__)


def cleanup_video_artifacts(resolution, settings) -> list[str]:
    removed: list[str] = []
    candidates = [
        settings.vk_cache_path,
        settings.vk_cache_path.with_suffix(".mp4.part"),
        Path("data_cache/vk_cache.mp4.part"),
    ]
    for candidate in candidates:
        try:
            if candidate.exists() and candidate.resolve() != settings.video_fallback.resolve():
                candidate.unlink()
                removed.append(str(candidate))
        except FileNotFoundError:
            continue
    return removed


def replace_detection_events(engine, events: pd.DataFrame) -> int:
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE core.detection_events"))
    events.to_sql("detection_events", engine, schema="core", if_exists="append", index=False, method="multi")
    return len(events)


def replace_tracks(engine, tracks: pd.DataFrame) -> int:
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE core.tracked_objects"))
    tracks.to_sql("tracked_objects", engine, schema="core", if_exists="append", index=False, method="multi")
    return len(tracks)


def replace_streaming_metrics(engine, metrics: pd.DataFrame) -> int:
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE dm.streaming_metrics"))
    metrics.to_sql("streaming_metrics", engine, schema="dm", if_exists="append", index=False, method="multi")
    return len(metrics)


def run() -> None:
    parser = argparse.ArgumentParser(description="Run streaming pipeline.")
    parser.add_argument("--source", default=None, help="Local file path or remote page/media URL.")
    parser.add_argument("--fallback-video", default=None, help="Local fallback video path.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)
    settings = get_settings()
    if args.source:
        settings.stream_source = args.source
    if args.fallback_video:
        settings.video_fallback = Path(args.fallback_video)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    engine = get_engine(settings)
    resolution = resolve_video_source(settings)
    config_snapshot = {
        "stream_source": settings.stream_source,
        "video_resolution": asdict(resolution),
        "window_minutes": settings.stream_window_minutes,
        "yolo_model_path": str(settings.yolo_model_path),
        "stream_target_fps": settings.stream_target_fps,
        "use_scene_profile": settings.use_scene_profile,
        "motion_roi_enabled": settings.motion_roi_enabled,
    }
    run_id = start_pipeline_run(
        engine,
        pipeline_name="traffic_streaming_pipeline",
        pipeline_mode="streaming",
        source_ref=resolution.resolved_uri,
        config_snapshot=config_snapshot,
    )
    rows_written = 0
    try:
        video_source_id = register_video_source(engine, resolution)
        artifacts_dir = settings.artifacts_dir / "streaming_samples"
        backend_name = "yolo_ultralytics"
        sample_frames: list[str] = []
        try:
            primary_source = (
                resolution.resolved_uri
                if resolution.source_type in {"remote_direct", "remote_page_direct"}
                else str(resolution.local_path)
            )
            detected_events, sample_frames = detect_and_track_video(
                settings=settings,
                source_path=primary_source,
                fps=resolution.fps,
                width=resolution.width,
                height=resolution.height,
                artifacts_dir=artifacts_dir,
            )
            if detected_events.empty:
                raise RuntimeError("YOLO detection returned zero events.")
            events = detected_events.copy()
            events["run_id"] = str(run_id)
            events["video_source_id"] = video_source_id
            events["camera_id"] = settings.stream_camera_id
            events["source_track_id"] = None
            events["direction_hint"] = None
            events["event_source"] = backend_name
            events = events[
                [
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
            ]
            tracks_source = events.rename(columns={"frame_ts": "detection_time"})
            tracks_source["x_cord_m"] = tracks_source["centroid_x"]
            tracks_source["y_cord_m"] = tracks_source["centroid_y"]
            tracks_source["direction"] = None
            tracks = build_tracks(
                tracks_source[
                    ["track_id", "vehicle_class", "detection_time", "x_cord_m", "y_cord_m", "direction"]
                ]
            )
        except Exception as yolo_exc:
            LOGGER.warning("Primary YOLO backend failed: %s", yolo_exc)
            try:
                LOGGER.warning("Retrying YOLO on local fallback video %s", settings.video_fallback)
                resolution.message = (
                    f"{resolution.message} Primary source failed; YOLO retried on local fallback video."
                )
                resolution.source_type = f"{resolution.source_type}_fallback_to_local"
                detected_events, sample_frames = detect_and_track_video(
                    settings=settings,
                    source_path=str(settings.video_fallback),
                    fps=30.0,
                    width=resolution.width,
                    height=resolution.height,
                    artifacts_dir=artifacts_dir,
                )
                if detected_events.empty:
                    raise RuntimeError("YOLO detection on local fallback returned zero events.")
                events = detected_events.copy()
                events["run_id"] = str(run_id)
                events["video_source_id"] = video_source_id
                events["camera_id"] = settings.stream_camera_id
                events["source_track_id"] = None
                events["direction_hint"] = None
                events["event_source"] = backend_name
                events = events[
                    [
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
                ]
                tracks_source = events.rename(columns={"frame_ts": "detection_time"})
                tracks_source["x_cord_m"] = tracks_source["centroid_x"]
                tracks_source["y_cord_m"] = tracks_source["centroid_y"]
                tracks_source["direction"] = None
                tracks = build_tracks(
                    tracks_source[
                        ["track_id", "vehicle_class", "detection_time", "x_cord_m", "y_cord_m", "direction"]
                    ]
                )
            except Exception as local_yolo_exc:
                LOGGER.warning(
                    "YOLO backend failed on both primary and local fallback sources, switching to preannotated fallback: %s",
                    local_yolo_exc,
                )
                backend_name = "preannotated_fallback"
                events_raw = load_preannotated_events(settings.tracking_fallback, resolution.fps)
                events = pd.DataFrame(
                    {
                        "run_id": str(run_id),
                        "video_source_id": video_source_id,
                        "camera_id": settings.stream_camera_id,
                        "frame_ts": events_raw["detection_time"],
                        "frame_no": events_raw["frame_no"],
                        "track_id": events_raw["track_key"],
                        "source_track_id": events_raw["track_id"],
                        "class_name_raw": events_raw["object_type"],
                        "vehicle_class": events_raw["vehicle_class"],
                        "confidence": None,
                        "centroid_x": events_raw["x_cord_m"],
                        "centroid_y": events_raw["y_cord_m"],
                        "bbox_x1": None,
                        "bbox_y1": None,
                        "bbox_x2": None,
                        "bbox_y2": None,
                        "direction_hint": events_raw["direction"],
                        "speed_proxy": events_raw["speed_proxy"],
                        "event_source": backend_name,
                    }
                )
                tracks = build_tracks(events_raw)
                sample_frames = []
        tracks["run_id"] = str(run_id)
        tracks["video_source_id"] = video_source_id
        tracks["camera_id"] = settings.stream_camera_id
        tracks["event_source"] = backend_name
        track_columns = [
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
        metrics = build_streaming_metrics(tracks, settings.stream_camera_id, settings.stream_window_minutes)
        metrics["run_id"] = str(run_id)
        metrics = metrics[
            [
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
        ]
        detection_count = replace_detection_events(engine, events)
        track_count = replace_tracks(engine, tracks[track_columns])
        metric_count = replace_streaming_metrics(engine, metrics)
        record_quality_check(
            engine,
            run_id,
            dataset_name="streaming_detection_events",
            check_name="streaming_backend",
            check_scope="pipeline",
            status="passed" if backend_name == "yolo_ultralytics" else "warning",
            actual_value=1,
            threshold_value=0,
            details_json={
                "backend": backend_name,
                "video_source_type": resolution.source_type,
                "sample_frames": sample_frames,
            },
        )
        record_quality_check(
            engine,
            run_id,
            dataset_name="streaming_detection_events",
            check_name="non_empty_detection_events",
            check_scope="dataset",
            status="passed" if detection_count > 0 else "failed",
            actual_value=detection_count,
            threshold_value=1,
        )
        rows_written = detection_count + track_count + metric_count
        removed_files = cleanup_video_artifacts(resolution, settings)
        if removed_files:
            record_quality_check(
                engine,
                run_id,
                dataset_name="streaming_video_cleanup",
                check_name="temporary_video_artifacts_removed",
                check_scope="pipeline",
                status="passed",
                actual_value=len(removed_files),
                threshold_value=1,
                details_json={"removed_files": removed_files},
            )
        finish_pipeline_run(engine, run_id, status="success", rows_written=rows_written)
        LOGGER.info(
            "Streaming pipeline completed: detections=%s tracks=%s metrics=%s",
            detection_count,
            track_count,
            metric_count,
        )
    except Exception as exc:
        finish_pipeline_run(engine, run_id, status="failed", rows_written=rows_written, error_text=str(exc))
        raise


if __name__ == "__main__":
    run()
