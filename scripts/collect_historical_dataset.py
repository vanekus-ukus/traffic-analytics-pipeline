from __future__ import annotations

import argparse
import copy
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import cv2
import pandas as pd

from traffic_analytics.config.settings import get_settings
from traffic_analytics.streaming.yolo_backend import detect_and_track_video


DIMENSIONS = {
    "car": (1.8, 4.6, "light car"),
    "truck": (2.5, 7.0, "truck"),
    "bus": (2.6, 10.5, "bus"),
    "motorcycle": (0.8, 2.1, "motorcycle"),
    "bicycle": (0.6, 1.8, "bicycle"),
    "person": (0.6, 0.6, "pedestrian"),
}

LOGGER = logging.getLogger(__name__)


def _format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "-"
    return ", ".join(f"{name}={value}" for name, value in sorted(counts.items()))


def _setup_logging(log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def _capture_stream(url: str, output_path: Path, duration_seconds: int) -> tuple[datetime, datetime]:
    ytdlp = Path(".venv/bin/yt-dlp")
    payload = json.loads(
        subprocess.check_output([str(ytdlp), "--dump-single-json", url], text=True)
    )
    requested = (payload.get("requested_downloads") or [{}])[0]
    stream_url = requested.get("url") or payload.get("url")
    headers = requested.get("http_headers") or payload.get("http_headers") or {}
    headers_text = "".join(f"{key}: {value}\r\n" for key, value in headers.items())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "error",
        "-progress",
        "pipe:1",
        "-headers",
        headers_text,
        "-i",
        stream_url,
        "-t",
        str(duration_seconds),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    LOGGER.info("capture started | source=%s | duration_seconds=%s | output=%s", url, duration_seconds, output_path)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    last_logged_second = -1
    assert process.stdout is not None
    for line in process.stdout:
        line = line.strip()
        if not line.startswith("out_time_ms="):
            continue
        out_time_ms = int(line.split("=", 1)[1] or 0)
        current_second = out_time_ms // 1_000_000
        if current_second >= 0 and current_second != last_logged_second and current_second % 15 == 0:
            last_logged_second = current_second
            LOGGER.info(
                "capture progress | elapsed=%ss | target=%ss | file=%s",
                current_second,
                duration_seconds,
                output_path.name,
            )
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)
    ended_at = datetime.now(timezone.utc)
    LOGGER.info("capture finished | elapsed_seconds=%s | output=%s", duration_seconds, output_path)
    return started_at, ended_at


def _probe_frame_shape(video_path: Path) -> tuple[int, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    capture.release()
    return width, height


def _project_coords(df: pd.DataFrame, frame_width: int, frame_height: int) -> pd.DataFrame:
    projected = df.copy()
    projected["x_cord_m"] = (projected["centroid_x"] / max(frame_width, 1)) * 200.0
    projected["y_cord_m"] = (projected["centroid_y"] / max(frame_height, 1)) * 120.0
    return projected


def _enrich_track_metrics(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.sort_values(["track_id", "frame_ts"]).copy()
    grouped = enriched.groupby("track_id")
    enriched["prev_x_m"] = grouped["x_cord_m"].shift(1)
    enriched["prev_y_m"] = grouped["y_cord_m"].shift(1)
    enriched["prev_ts"] = grouped["frame_ts"].shift(1)
    dt = (enriched["frame_ts"] - enriched["prev_ts"]).dt.total_seconds()
    dx = enriched["x_cord_m"] - enriched["prev_x_m"]
    dy = enriched["y_cord_m"] - enriched["prev_y_m"]
    step_m = ((dx.pow(2) + dy.pow(2)) ** 0.5).where(dt > 0)
    enriched["step_m"] = step_m.fillna(0.0)
    enriched["speed_km_h"] = ((step_m / dt) * 3.6).where(dt > 0)
    enriched["total_kms"] = grouped["step_m"].cumsum() / 1000.0
    enriched["dx"] = dx.fillna(0.0)
    enriched["direction"] = enriched["dx"].apply(
        lambda value: "positive_x" if value > 0.1 else ("negative_x" if value < -0.1 else "stationary")
    )
    return enriched


def _export_sample_format(
    events: pd.DataFrame,
    video_path: Path,
    csv_path: Path,
    recording_at: datetime,
    recording_end: datetime,
    video_title: str,
) -> None:
    frame_width, frame_height = _probe_frame_shape(video_path)
    exported = _project_coords(events, frame_width, frame_height)
    exported = _enrich_track_metrics(exported)

    track_summary = (
        exported.groupby("track_id")
        .agg(start_time=("frame_ts", "min"), end_time=("frame_ts", "max"))
        .reset_index()
    )
    exported = exported.merge(track_summary, on="track_id", how="left")

    track_ids = {track_id: idx + 1 for idx, track_id in enumerate(sorted(exported["track_id"].unique()))}
    exported["tracker_id"] = exported["track_id"].map(track_ids)
    exported["numeric_track_id"] = exported["track_id"].map(track_ids)

    exported["object_type"] = exported["vehicle_class"].map(
        lambda cls: DIMENSIONS.get(cls, (1.0, 1.0, cls))[2]
    )
    exported["width"] = exported["vehicle_class"].map(lambda cls: DIMENSIONS.get(cls, (1.0, 1.0, cls))[0])
    exported["length"] = exported["vehicle_class"].map(lambda cls: DIMENSIONS.get(cls, (1.0, 1.0, cls))[1])

    result = pd.DataFrame(
        {
            "video_id": 1,
            "title": video_title,
            "path": str(video_path),
            "recording_at": recording_at.isoformat(),
            "recording_end": recording_end.isoformat(),
            "track_id": exported["numeric_track_id"],
            "tracker_id": exported["tracker_id"],
            "start_time": exported["start_time"].astype(str),
            "end_time": exported["end_time"].astype(str),
            "object_type": exported["object_type"],
            "width": exported["width"],
            "length": exported["length"],
            "detection_time": exported["frame_ts"].astype(str),
            "x_cord_m": exported["x_cord_m"].round(5),
            "y_cord_m": exported["y_cord_m"].round(5),
            "total_kms": exported["total_kms"].round(5),
            "speed_km_h": exported["speed_km_h"].round(5),
            "direction": exported["direction"],
        }
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(csv_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--duration-seconds", type=int, default=1800)
    parser.add_argument("--output-dir", default="collected_data")
    parser.add_argument("--historical-fps", type=int, default=5)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--progress-interval-seconds", type=int, default=15)
    args = parser.parse_args()

    _setup_logging(Path(args.log_file) if args.log_file else None)

    started_at = datetime.now()
    stamp = started_at.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    video_path = output_dir / f"video_capture_{stamp}.mp4"
    csv_path = output_dir / f"tracking_capture_{stamp}.csv"
    artifacts_dir = output_dir / f"artifacts_{stamp}"

    recording_at, recording_end = _capture_stream(args.source, video_path, args.duration_seconds)

    settings = copy.deepcopy(get_settings())
    settings.stream_target_fps = args.historical_fps
    settings.stream_max_seconds = max(args.duration_seconds, 1)
    settings.artifacts_dir = artifacts_dir
    LOGGER.info(
        "processing started | video=%s | fps=%s | artifacts=%s",
        video_path,
        settings.stream_target_fps,
        artifacts_dir,
    )

    state = {"last_logged_second": -1}

    def progress_callback(payload: dict[str, object]) -> None:
        source_seconds = int(float(payload["source_seconds"]))
        interval = max(args.progress_interval_seconds, 1)
        if source_seconds == state["last_logged_second"] or source_seconds % interval != 0:
            return
        state["last_logged_second"] = source_seconds
        LOGGER.info(
            "processing progress | t=%ss | processed_frames=%s | accepted_events=%s | unique_tracks=%s | classes=%s",
            source_seconds,
            payload["processed_frames"],
            payload["accepted_events"],
            payload["unique_tracks_seen"],
            _format_counts(payload["class_totals"]),
        )

    events, _ = detect_and_track_video(
        settings=settings,
        source_path=str(video_path),
        fps=None,
        width=None,
        height=None,
        artifacts_dir=artifacts_dir,
        base_time=recording_at,
        progress_callback=progress_callback,
    )
    _export_sample_format(
        events=events,
        video_path=video_path,
        csv_path=csv_path,
        recording_at=recording_at,
        recording_end=recording_end,
        video_title=video_path.stem,
    )
    class_counts = events["vehicle_class"].value_counts().to_dict() if not events.empty else {}
    track_count = events["track_id"].nunique() if not events.empty else 0
    LOGGER.info(
        "processing finished | events=%s | tracks=%s | classes=%s",
        len(events),
        track_count,
        _format_counts(class_counts),
    )
    LOGGER.info("video saved | %s", video_path)
    LOGGER.info("csv saved | %s", csv_path)
    print(video_path)
    print(csv_path)
    print(f"events={len(events)}")


if __name__ == "__main__":
    main()
