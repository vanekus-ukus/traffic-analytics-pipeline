from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import logging
from pathlib import Path
import shutil
import subprocess
import time
from uuid import uuid4

import pandas as pd

from traffic_analytics.common.logging_utils import configure_logging
from traffic_analytics.config.settings import get_settings
from traffic_analytics.db.engine import get_engine
from traffic_analytics.db.runtime import finish_pipeline_run, record_quality_check, start_pipeline_run
from traffic_analytics.streaming.storage import (
    append_detection_events,
    append_tracks,
    build_tracks_from_events,
    prepare_events,
    replace_run_metrics,
    summarize_detected_tracks,
)
from traffic_analytics.streaming.track_stitcher import TrackStitchState, stitch_segment_tracks
from traffic_analytics.streaming.video_source import (
    MediaRequest,
    VideoResolution,
    register_video_source,
    resolve_validated_media_request,
)
from traffic_analytics.streaming.yolo_backend import detect_and_track_video, export_annotated_video

LOGGER = logging.getLogger(__name__)


def _format_counts(series: pd.Series) -> str:
    if series.empty:
        return "-"
    counts = series.value_counts().sort_index().to_dict()
    return ", ".join(f"{name}={value}" for name, value in counts.items())


def probe_segment(path: Path) -> tuple[float | None, float | None, int | None, int | None]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=0",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        return None, None, None, None

    duration = None
    fps = None
    width = None
    height = None
    for line in result.stdout.splitlines():
        if line.startswith("duration="):
            try:
                duration = float(line.split("=", 1)[1])
            except ValueError:
                pass
        elif line.startswith("width="):
            width = int(line.split("=", 1)[1])
        elif line.startswith("height="):
            height = int(line.split("=", 1)[1])
        elif line.startswith("r_frame_rate="):
            rate = line.split("=", 1)[1]
            if "/" in rate:
                num, den = rate.split("/", 1)
                if den != "0":
                    fps = float(num) / float(den)
    return duration, fps, width, height


def build_session_video_source(source: str, segment_dir: Path) -> VideoResolution:
    return VideoResolution(
        requested_uri=source,
        resolved_uri=source,
        local_path=segment_dir,
        source_type="live_segmented",
        status="live_session_started",
        message="Segmented live capture session.",
        duration_sec=None,
        fps=None,
        width=None,
        height=None,
    )


def capture_remote_segment(
    media_request: MediaRequest,
    output_part: Path,
    segment_seconds: int,
    timeout_seconds: int,
) -> Path | None:
    output_part.parent.mkdir(parents=True, exist_ok=True)
    headers_text = "".join(f"{key}: {value}\r\n" for key, value in media_request.headers.items())
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-headers",
        headers_text,
        "-i",
        media_request.url,
        "-t",
        str(segment_seconds),
        "-c",
        "copy",
        "-f",
        "mpegts",
        f"file:{output_part}",
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)

    if not output_part.exists() or output_part.stat().st_size <= 1024:
        return None

    final_path = output_part.with_suffix("")
    output_part.replace(final_path)
    duration, _, _, _ = probe_segment(final_path)
    if duration is None or duration <= 0.5:
        final_path.unlink(missing_ok=True)
        return None
    return final_path


def capture_local_segment(source: str, output_path: Path, offset_seconds: int, segment_seconds: int) -> Path | None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-ss",
        str(offset_seconds),
        "-i",
        source,
        "-t",
        str(segment_seconds),
        "-c",
        "copy",
        "-f",
        "mpegts",
        str(output_path),
    ]
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not output_path.exists() or output_path.stat().st_size <= 1024:
        return None
    duration, _, _, _ = probe_segment(output_path)
    if duration is None or duration <= 0.5:
        output_path.unlink(missing_ok=True)
        return None
    return output_path


def is_local_source(source: str) -> bool:
    return Path(source).exists()


def run() -> None:
    parser = argparse.ArgumentParser(description="Run segmented live traffic detection.")
    parser.add_argument("--source", default=None, help="Page URL, direct media URL, or local video file.")
    parser.add_argument("--duration-seconds", type=int, default=None)
    parser.add_argument("--segment-seconds", type=int, default=None)
    parser.add_argument("--keep-segments", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    settings = get_settings()
    if args.source:
        settings.stream_source = args.source
    if args.duration_seconds is not None:
        settings.live_runtime_seconds = args.duration_seconds
    if args.segment_seconds is not None:
        settings.live_segment_seconds = args.segment_seconds
    if args.keep_segments:
        settings.live_keep_segments = True
    settings.stream_max_seconds = settings.live_segment_seconds

    segment_root = settings.live_segment_dir / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    segment_root.mkdir(parents=True, exist_ok=True)
    segment_data_dir = segment_root / "segments"
    segment_data_dir.mkdir(parents=True, exist_ok=True)
    sample_root = segment_root / "samples"
    sample_root.mkdir(parents=True, exist_ok=True)
    annotated_root = segment_root / "annotated"
    annotated_root.mkdir(parents=True, exist_ok=True)

    engine = None
    db_enabled = True
    try:
        engine = get_engine(settings)
    except Exception as exc:
        db_enabled = False
        LOGGER.warning("Database connection disabled for this run: %s", exc)
    config_snapshot = {
        "stream_source": settings.stream_source,
        "live_segment_seconds": settings.live_segment_seconds,
        "live_runtime_seconds": settings.live_runtime_seconds,
        "stream_target_fps": settings.stream_target_fps,
        "motion_roi_enabled": settings.motion_roi_enabled,
        "use_scene_profile": settings.use_scene_profile,
        "segment_root": str(segment_root),
    }
    if db_enabled and engine is not None:
        try:
            run_id = start_pipeline_run(
                engine,
                pipeline_name="traffic_live_stream",
                pipeline_mode="live",
                source_ref=settings.stream_source,
                config_snapshot=config_snapshot,
            )
        except Exception as exc:
            db_enabled = False
            LOGGER.warning("Database writes disabled for this run: %s", exc)
            run_id = uuid4()
    else:
        run_id = uuid4()
    rows_written = 0
    total_detections = 0
    total_tracks = 0
    segments_processed = 0
    sample_frames_all: list[str] = []
    cumulative_tracks: list[pd.DataFrame] = []
    stitch_state = TrackStitchState()
    consecutive_failures = 0
    active_media_request: MediaRequest | None = None
    session_time_anchor = datetime.now(timezone.utc)
    start_monotonic = time.monotonic()

    try:
        video_source_id = (
            register_video_source(engine, build_session_video_source(settings.stream_source, segment_root))
            if db_enabled and engine is not None
            else 0
        )
        segment_index = 0
        local_offset_seconds = 0
        while time.monotonic() - start_monotonic < settings.live_runtime_seconds:
            segment_index += 1
            segment_name = f"segment_{segment_index:06d}.ts"
            segment_path = segment_data_dir / segment_name
            captured_path: Path | None

            if is_local_source(settings.stream_source):
                segment_base_time = session_time_anchor + pd.Timedelta(seconds=local_offset_seconds)
                captured_path = capture_local_segment(
                    settings.stream_source,
                    segment_path,
                    offset_seconds=local_offset_seconds,
                    segment_seconds=settings.live_segment_seconds,
                )
                local_offset_seconds += settings.live_segment_seconds
            else:
                segment_base_time = datetime.now(timezone.utc)
                if active_media_request is None or settings.live_resolve_each_cycle:
                    active_media_request = resolve_validated_media_request(settings.stream_source)
                if active_media_request is None:
                    consecutive_failures += 1
                    LOGGER.warning("No validated stream candidate for source %s", settings.stream_source)
                    if consecutive_failures >= settings.live_max_consecutive_failures:
                        time.sleep(settings.live_retry_backoff_seconds)
                    continue
                captured_path = capture_remote_segment(
                    active_media_request,
                    output_part=segment_path.with_suffix(".ts.part"),
                    segment_seconds=settings.live_segment_seconds,
                    timeout_seconds=settings.live_capture_timeout_seconds,
                )

            if captured_path is None:
                consecutive_failures += 1
                LOGGER.warning("Skipping segment %s: capture failed or produced no usable media", segment_index)
                active_media_request = None
                if consecutive_failures >= settings.live_max_consecutive_failures:
                    LOGGER.warning(
                        "Reached %s consecutive capture failures; backing off for %ss",
                        consecutive_failures,
                        settings.live_retry_backoff_seconds,
                    )
                    time.sleep(settings.live_retry_backoff_seconds)
                continue
            consecutive_failures = 0

            duration, fps, width, height = probe_segment(captured_path)
            if duration is None or duration <= 0.5:
                LOGGER.warning("Skipping segment %s: ffprobe found no usable duration", segment_index)
                captured_path.unlink(missing_ok=True)
                continue

            segment_samples_dir = sample_root / f"segment_{segment_index:06d}"
            detected_events, sample_frames = detect_and_track_video(
                settings=settings,
                source_path=str(captured_path),
                fps=fps,
                width=width,
                height=height,
                artifacts_dir=segment_samples_dir,
                base_time=segment_base_time,
            )
            if detected_events.empty:
                LOGGER.info("Segment %s produced zero filtered detections", segment_index)
                if not settings.live_keep_segments:
                    captured_path.unlink(missing_ok=True)
                continue

            stitched_count = 0
            if settings.live_track_stitch_enabled:
                local_tracks = summarize_detected_tracks(detected_events)
                track_mapping, stitched_count = stitch_segment_tracks(local_tracks, stitch_state, settings)
                if track_mapping:
                    detected_events = detected_events.copy()
                    detected_events["track_id"] = detected_events["track_id"].map(
                        lambda value: track_mapping.get(str(value), str(value))
                    )
            else:
                detected_events = detected_events.copy()
                detected_events["track_id"] = detected_events["track_id"].map(
                    lambda value: f"seg{segment_index:06d}_{value}"
                )

            events = prepare_events(
                detected_events=detected_events,
                run_id=run_id,
                video_source_id=video_source_id,
                camera_id=settings.stream_camera_id,
                event_source="yolo_ultralytics_live",
            )
            tracks = build_tracks_from_events(
                events=events,
                run_id=run_id,
                video_source_id=video_source_id,
                camera_id=settings.stream_camera_id,
                event_source="yolo_ultralytics_live",
            )

            detection_count = append_detection_events(engine, events) if db_enabled and engine is not None else len(events)
            track_count = append_tracks(engine, tracks) if db_enabled and engine is not None else len(tracks)
            if not tracks.empty:
                cumulative_tracks.append(tracks.copy())
            metric_count = (
                replace_run_metrics(
                    engine,
                    run_id=run_id,
                    tracks_for_run=pd.concat(cumulative_tracks, ignore_index=True) if cumulative_tracks else pd.DataFrame(),
                    settings=settings,
                )
                if db_enabled and engine is not None
                else 0
            )

            total_detections += detection_count
            total_tracks += track_count
            rows_written += detection_count + track_count
            segments_processed += 1
            sample_frames_all.extend(sample_frames[:2])

            segment_detection_counts = _format_counts(events["vehicle_class"])
            segment_track_counts = _format_counts(tracks["vehicle_class"]) if not tracks.empty else "-"
            segment_ts_start = events["frame_ts"].min()
            segment_ts_end = events["frame_ts"].max()
            annotated_segment_path = annotated_root / f"segment_{segment_index:06d}.mp4"
            export_annotated_video(
                source_path=str(captured_path),
                events=detected_events,
                output_path=annotated_segment_path,
                fps=fps,
            )
            cumulative_track_frame = (
                pd.concat(cumulative_tracks, ignore_index=True) if cumulative_tracks else pd.DataFrame()
            )
            cumulative_track_counts = (
                _format_counts(cumulative_track_frame["vehicle_class"])
                if not cumulative_track_frame.empty
                else "-"
            )

            LOGGER.info(
                "segment %s | time=%s..%s | duration=%.2fs | detections=%s | tracks=%s | metrics=%s",
                segment_index,
                segment_ts_start,
                segment_ts_end,
                duration,
                detection_count,
                track_count,
                metric_count,
            )
            LOGGER.info(
                "segment %s classes | detections=%s | tracks=%s",
                segment_index,
                segment_detection_counts,
                segment_track_counts,
            )
            LOGGER.info("segment %s annotated_video | %s", segment_index, annotated_segment_path)
            LOGGER.info(
                "cumulative | segments=%s | detections=%s | tracks=%s | track_classes=%s",
                segments_processed,
                total_detections,
                total_tracks,
                cumulative_track_counts,
            )
            if stitched_count:
                LOGGER.info("Stitched %s tracks across segment boundary for segment %s", stitched_count, segment_index)

            if not settings.live_keep_segments:
                captured_path.unlink(missing_ok=True)

        if db_enabled and engine is not None:
            record_quality_check(
                engine,
                run_id,
                dataset_name="live_stream_detection_events",
                check_name="live_segments_processed",
                check_scope="pipeline",
                status="passed" if segments_processed > 0 else "failed",
                actual_value=segments_processed,
                threshold_value=1,
                details_json={
                    "sample_frames": sample_frames_all[:10],
                    "segment_root": str(segment_root),
                    "active_stitched_tracks": int(len(stitch_state.active_tracks)),
                },
            )
            record_quality_check(
                engine,
                run_id,
                dataset_name="live_stream_source",
                check_name="validated_stream_selected",
                check_scope="pipeline",
                status="passed" if is_local_source(settings.stream_source) or active_media_request else "failed",
                actual_value=1 if is_local_source(settings.stream_source) or active_media_request else 0,
                threshold_value=1,
                details_json={
                    "stream_url": active_media_request.url if active_media_request else None,
                    "format_id": active_media_request.format_id if active_media_request else None,
                    "width": active_media_request.width if active_media_request else None,
                    "height": active_media_request.height if active_media_request else None,
                    "fps": active_media_request.fps if active_media_request else None,
                    "protocol": active_media_request.protocol if active_media_request else None,
                },
            )
            record_quality_check(
                engine,
                run_id,
                dataset_name="live_stream_detection_events",
                check_name="non_empty_detection_events",
                check_scope="dataset",
                status="passed" if total_detections > 0 else "failed",
                actual_value=total_detections,
                threshold_value=1,
            )
            record_quality_check(
                engine,
                run_id,
                dataset_name="live_stream_capture",
                check_name="consecutive_capture_failures_max",
                check_scope="pipeline",
                status="passed" if consecutive_failures < settings.live_max_consecutive_failures else "warning",
                actual_value=consecutive_failures,
                threshold_value=settings.live_max_consecutive_failures,
            )
            finish_pipeline_run(engine, run_id, status="success", rows_written=rows_written)
        if not settings.live_keep_segments and segment_data_dir.exists():
            shutil.rmtree(segment_data_dir, ignore_errors=True)
        annotated_segments = sorted(annotated_root.glob("segment_*.mp4"))
        if annotated_segments:
            concat_list_path = segment_root / "annotated_segments.txt"
            concat_list_path.write_text(
                "\n".join(f"file '{path.resolve()}'" for path in annotated_segments),
                encoding="utf-8",
            )
            final_annotated_path = segment_root / "annotated_full.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_list_path),
                    "-c",
                    "copy",
                    str(final_annotated_path),
                ],
                check=False,
            )
            if final_annotated_path.exists():
                LOGGER.info("annotated full video | %s", final_annotated_path)
        LOGGER.info(
            "Live stream run completed: segments=%s detections=%s tracks=%s db_enabled=%s",
            segments_processed,
            total_detections,
            total_tracks,
            db_enabled,
        )
    except Exception as exc:
        if db_enabled and engine is not None:
            finish_pipeline_run(engine, run_id, status="failed", rows_written=rows_written, error_text=str(exc))
        raise


if __name__ == "__main__":
    run()
