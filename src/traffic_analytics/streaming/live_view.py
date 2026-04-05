from __future__ import annotations

import argparse
from collections import Counter
from collections import deque
from dataclasses import dataclass, field
import logging
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
from ultralytics import YOLO

from traffic_analytics.common.logging_utils import configure_logging
from traffic_analytics.common.vehicle_mapping import map_vehicle_class
from traffic_analytics.config.settings import get_settings
from traffic_analytics.streaming.video_source import MediaRequest, resolve_validated_media_request
from traffic_analytics.streaming.yolo_backend import COCO_TARGETS, VEHICLE_CLASSES

LOGGER = logging.getLogger(__name__)


@dataclass
class LiveTrackState:
    global_track_id: str
    local_track_id: str | None
    first_seen: float
    last_seen: float
    prev_centroid_x: float
    prev_centroid_y: float
    centroid_x: float
    centroid_y: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    touches_edge: bool
    class_votes: Counter[str] = field(default_factory=Counter)
    counted: bool = False
    suppressed: bool = False

    @property
    def stable_class(self) -> str:
        return self.class_votes.most_common(1)[0][0] if self.class_votes else "unknown"


def _apply_scene_preset(settings, preset: str) -> None:
    if preset == "fast_road":
        settings.stream_target_fps = max(settings.stream_target_fps, 15)
        settings.yolo_imgsz = min(settings.yolo_imgsz, 768)
        settings.yolo_confidence = min(settings.yolo_confidence, 0.20)
        settings.edge_border_ratio = max(settings.edge_border_ratio, 0.08)
        settings.upper_zone_vehicle_bottom_ratio = min(settings.upper_zone_vehicle_bottom_ratio, 0.15)


@dataclass
class ViewerTuning:
    overlap_iou_threshold: float = 0.55
    track_stale_seconds: float = 3.0
    stitch_gap_seconds: float = 1.5
    stitch_distance_px: float = 120.0
    min_track_hits: int = 2
    edge_min_track_hits: int = 1
    static_duration_seconds: float = 2.0
    static_movement_px: float = 4.0
    suppression_padding_px: float = 20.0


def _class_family(vehicle_class: str) -> str:
    if vehicle_class in {"car", "truck", "bus", "motorcycle"}:
        return "vehicle"
    return vehicle_class


def _iou(a: dict[str, object], b: dict[str, object]) -> float:
    ax1, ay1, ax2, ay2 = float(a["bbox_x1"]), float(a["bbox_y1"]), float(a["bbox_x2"]), float(a["bbox_y2"])
    bx1, by1, bx2, by2 = float(b["bbox_x1"]), float(b["bbox_y1"]), float(b["bbox_x2"]), float(b["bbox_y2"])
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _suppress_overlaps(rows: list[dict[str, object]], overlap_iou_threshold: float) -> list[dict[str, object]]:
    ordered = sorted(rows, key=lambda row: float(row["confidence"]), reverse=True)
    kept: list[dict[str, object]] = []
    for row in ordered:
        drop = False
        for kept_row in kept:
            same_family = _class_family(str(row["vehicle_class"])) == _class_family(str(kept_row["vehicle_class"]))
            if same_family and _iou(row, kept_row) >= overlap_iou_threshold:
                drop = True
                break
        if not drop:
            kept.append(row)
    return kept


def _stable_track_counts(track_states: dict[str, LiveTrackState]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for state in track_states.values():
        if state.counted:
            counts[state.stable_class] += 1
    return counts


def _row_center_in_zone(row: dict[str, object], zone: dict[str, float]) -> bool:
    cx = float(row["centroid_x"])
    cy = float(row["centroid_y"])
    return zone["x1"] <= cx <= zone["x2"] and zone["y1"] <= cy <= zone["y2"]


def _apply_suppression_zones(rows: list[dict[str, object]], suppression_zones: list[dict[str, float]]) -> list[dict[str, object]]:
    if not suppression_zones:
        return rows
    filtered: list[dict[str, object]] = []
    for row in rows:
        if any(_row_center_in_zone(row, zone) for zone in suppression_zones):
            continue
        filtered.append(row)
    return filtered


def _assign_global_track(
    row: dict[str, object],
    track_states: dict[str, LiveTrackState],
    local_to_global: dict[str, str],
    now_ts: float,
    next_track_index: int,
    max_gap_seconds: float,
    max_distance_px: float,
) -> tuple[str, int]:
    local_track_id = row.get("track_id")
    if local_track_id and local_track_id in local_to_global:
        return local_to_global[local_track_id], next_track_index

    family = _class_family(str(row["vehicle_class"]))
    cx = float(row["centroid_x"])
    cy = float(row["centroid_y"])
    best_id: str | None = None
    best_distance: float | None = None
    for global_id, state in track_states.items():
        if now_ts - state.last_seen > max_gap_seconds:
            continue
        if _class_family(state.stable_class) != family:
            continue
        distance = ((state.centroid_x - cx) ** 2 + (state.centroid_y - cy) ** 2) ** 0.5
        if distance > max_distance_px:
            continue
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_id = global_id

    if best_id is None:
        best_id = f"live_view_{next_track_index:08d}"
        next_track_index += 1
    if local_track_id:
        local_to_global[local_track_id] = best_id
    return best_id, next_track_index


def _update_track_states(
    rows: list[dict[str, object]],
    track_states: dict[str, LiveTrackState],
    local_to_global: dict[str, str],
    next_track_index: int,
    now_ts: float,
    tuning: ViewerTuning,
    suppression_zones: list[dict[str, float]],
) -> int:
    stale_ids = [track_id for track_id, state in track_states.items() if now_ts - state.last_seen > tuning.track_stale_seconds]
    for track_id in stale_ids:
        del track_states[track_id]
    stale_local = [local_id for local_id, global_id in local_to_global.items() if global_id not in track_states]
    for local_id in stale_local:
        del local_to_global[local_id]

    for row in rows:
        global_id, next_track_index = _assign_global_track(
            row,
            track_states,
            local_to_global,
            now_ts,
            next_track_index,
            max_gap_seconds=tuning.stitch_gap_seconds,
            max_distance_px=tuning.stitch_distance_px,
        )
        state = track_states.get(global_id)
        if state is None:
            state = LiveTrackState(
                global_track_id=global_id,
                local_track_id=row.get("track_id"),
                first_seen=now_ts,
                last_seen=now_ts,
                prev_centroid_x=float(row["centroid_x"]),
                prev_centroid_y=float(row["centroid_y"]),
                centroid_x=float(row["centroid_x"]),
                centroid_y=float(row["centroid_y"]),
                bbox_x1=float(row["bbox_x1"]),
                bbox_y1=float(row["bbox_y1"]),
                bbox_x2=float(row["bbox_x2"]),
                bbox_y2=float(row["bbox_y2"]),
                touches_edge=bool(row.get("touches_edge")),
            )
            track_states[global_id] = state
        state.local_track_id = row.get("track_id")
        state.last_seen = now_ts
        state.prev_centroid_x = state.centroid_x
        state.prev_centroid_y = state.centroid_y
        state.centroid_x = float(row["centroid_x"])
        state.centroid_y = float(row["centroid_y"])
        state.bbox_x1 = float(row["bbox_x1"])
        state.bbox_y1 = float(row["bbox_y1"])
        state.bbox_x2 = float(row["bbox_x2"])
        state.bbox_y2 = float(row["bbox_y2"])
        state.touches_edge = state.touches_edge or bool(row.get("touches_edge"))
        state.class_votes[str(row["vehicle_class"])] += 1
        row["stable_track_id"] = global_id
        row["vehicle_class"] = state.stable_class

        hits = sum(state.class_votes.values())
        if not state.counted:
            if hits >= tuning.min_track_hits or (state.touches_edge and hits >= tuning.edge_min_track_hits):
                state.counted = True
        duration = state.last_seen - state.first_seen
        movement = ((state.centroid_x - state.prev_centroid_x) ** 2 + (state.centroid_y - state.prev_centroid_y) ** 2) ** 0.5
        if (
            not state.suppressed
            and state.counted
            and _class_family(state.stable_class) == "vehicle"
            and duration >= tuning.static_duration_seconds
            and movement <= tuning.static_movement_px
        ):
            suppression_zones.append(
                {
                    "x1": max(0.0, state.bbox_x1 - tuning.suppression_padding_px),
                    "y1": max(0.0, state.bbox_y1 - tuning.suppression_padding_px),
                    "x2": state.bbox_x2 + tuning.suppression_padding_px,
                    "y2": state.bbox_y2 + tuning.suppression_padding_px,
                }
            )
            state.suppressed = True
    return next_track_index


def _build_ffmpeg_stream(
    request: MediaRequest,
    target_fps: int,
    output_width: int,
    output_height: int,
) -> subprocess.Popen[bytes]:
    headers_text = "".join(f"{key}: {value}\r\n" for key, value in request.headers.items())
    vf = f"fps={max(target_fps, 1)},scale={output_width}:{output_height}"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-headers",
        headers_text,
        "-i",
        request.url,
        "-an",
        "-vf",
        vf,
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def _build_ffplay_sink(output_width: int, output_height: int, fps: int) -> subprocess.Popen[bytes]:
    cmd = [
        "ffplay",
        "-hide_banner",
        "-loglevel",
        "error",
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{output_width}x{output_height}",
        "-framerate",
        str(max(fps, 1)),
        "-window_title",
        "traffic_live_view",
        "-",
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _draw_frame(
    frame: np.ndarray,
    rows: list[dict[str, object]],
    frame_class_counts: Counter[str],
    active_track_counts: Counter[str],
    lifetime_track_counts: Counter[str],
    recent_entry_counts: Counter[str],
    model_name: str,
) -> np.ndarray:
    output = frame.copy()
    for row in rows:
        x1, y1, x2, y2 = map(int, [row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]])
        label = f"{row['vehicle_class']} {float(row['confidence']):.2f}"
        color = (0, 220, 0)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(output, (x1, max(0, y1 - text_h - 8)), (x1 + text_w + 6, y1), color, -1)
        cv2.putText(
            output,
            label,
            (x1 + 3, max(12, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    panel_width = 360
    canvas = np.zeros((output.shape[0], output.shape[1] + panel_width, 3), dtype=np.uint8)
    canvas[:, : output.shape[1], :] = output
    canvas[:, output.shape[1] :, :] = (25, 25, 25)

    frame_stats = ", ".join(f"{name}={value}" for name, value in sorted(frame_class_counts.items())) or "-"
    active_stats = ", ".join(f"{name}={value}" for name, value in sorted(active_track_counts.items())) or "-"
    lifetime_stats = ", ".join(f"{name}={value}" for name, value in sorted(lifetime_track_counts.items())) or "-"
    recent_stats = ", ".join(f"{name}={value}" for name, value in sorted(recent_entry_counts.items())) or "-"

    x0 = output.shape[1] + 20
    lines = [
        f"model={model_name}",
        f"frame_counts: {frame_stats}",
        f"active_tracks: {active_stats}",
        f"lifetime_tracks: {lifetime_stats}",
        f"new_last_10s: {recent_stats}",
    ]
    for idx, text in enumerate(lines):
        y = 40 + idx * 34
        cv2.putText(canvas, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return canvas


def _rendered_frame_size(frame_width: int, frame_height: int) -> tuple[int, int]:
    panel_width = 360
    return frame_width + panel_width, frame_height


def _filter_rows(
    rows: list[dict[str, object]],
    frame_width: int,
    frame_height: int,
    settings,
) -> list[dict[str, object]]:
    filtered: list[dict[str, object]] = []
    for row in rows:
        vehicle_class = str(row["vehicle_class"])
        x1 = float(row["bbox_x1"])
        y1 = float(row["bbox_y1"])
        x2 = float(row["bbox_x2"])
        y2 = float(row["bbox_y2"])
        bbox_w = max(1.0, x2 - x1)
        bbox_h = max(1.0, y2 - y1)
        bbox_area_ratio = max(0.0, ((x2 - x1) * (y2 - y1)) / max(frame_width * frame_height, 1))
        bbox_aspect_ratio = bbox_w / bbox_h
        bbox_bottom_ratio = y2 / max(frame_height, 1)
        if bbox_area_ratio <= 0 or bbox_area_ratio > settings.max_bbox_area_ratio:
            continue
        if vehicle_class in VEHICLE_CLASSES:
            if (
                bbox_aspect_ratio < settings.vehicle_min_aspect_ratio
                or bbox_aspect_ratio > settings.vehicle_max_aspect_ratio
            ):
                continue
            if bbox_bottom_ratio < settings.upper_zone_vehicle_bottom_ratio:
                continue
        filtered.append(row)
    return filtered


def run() -> None:
    parser = argparse.ArgumentParser(description="Open live preview with detections over the incoming stream.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--display-height", type=int, default=720)
    parser.add_argument("--max-seconds", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--renderer", choices=["auto", "opencv", "ffplay"], default="auto")
    parser.add_argument("--scene-preset", choices=["default", "fast_road"], default="default")
    parser.add_argument("--target-fps", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--overlap-iou", type=float, default=None)
    parser.add_argument("--track-stale-seconds", type=float, default=None)
    parser.add_argument("--stitch-gap-seconds", type=float, default=None)
    parser.add_argument("--stitch-distance-px", type=float, default=None)
    parser.add_argument("--min-track-hits", type=int, default=None)
    parser.add_argument("--edge-min-track-hits", type=int, default=None)
    parser.add_argument("--static-duration-seconds", type=float, default=None)
    parser.add_argument("--static-movement-px", type=float, default=None)
    parser.add_argument("--suppression-padding-px", type=float, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    settings = get_settings()
    _apply_scene_preset(settings, args.scene_preset)
    if args.target_fps is not None:
        settings.stream_target_fps = args.target_fps
    if args.imgsz is not None:
        settings.yolo_imgsz = args.imgsz
    if args.confidence is not None:
        settings.yolo_confidence = args.confidence
    tuning = ViewerTuning()
    if args.scene_preset == "fast_road":
        tuning.track_stale_seconds = 4.0
        tuning.stitch_gap_seconds = 2.5
        tuning.stitch_distance_px = 220.0
        tuning.min_track_hits = 2
        tuning.edge_min_track_hits = 1
        tuning.static_duration_seconds = 2.0
        tuning.static_movement_px = 4.0
        tuning.overlap_iou_threshold = 0.60
    if args.overlap_iou is not None:
        tuning.overlap_iou_threshold = args.overlap_iou
    if args.track_stale_seconds is not None:
        tuning.track_stale_seconds = args.track_stale_seconds
    if args.stitch_gap_seconds is not None:
        tuning.stitch_gap_seconds = args.stitch_gap_seconds
    if args.stitch_distance_px is not None:
        tuning.stitch_distance_px = args.stitch_distance_px
    if args.min_track_hits is not None:
        tuning.min_track_hits = args.min_track_hits
    if args.edge_min_track_hits is not None:
        tuning.edge_min_track_hits = args.edge_min_track_hits
    if args.static_duration_seconds is not None:
        tuning.static_duration_seconds = args.static_duration_seconds
    if args.static_movement_px is not None:
        tuning.static_movement_px = args.static_movement_px
    if args.suppression_padding_px is not None:
        tuning.suppression_padding_px = args.suppression_padding_px
    request = resolve_validated_media_request(args.source)
    if request is None:
        raise RuntimeError(f"Unable to resolve a playable stream from {args.source}")

    source_width = int(request.width or 1280)
    source_height = int(request.height or 720)
    if source_height > args.display_height:
        output_height = int(args.display_height)
        output_width = int(round(source_width * (output_height / source_height)))
    else:
        output_width = source_width
        output_height = source_height
    output_width = max(2, output_width - (output_width % 2))
    output_height = max(2, output_height - (output_height % 2))
    rendered_width, rendered_height = _rendered_frame_size(output_width, output_height)

    model = YOLO(str(settings.yolo_model_path))
    process = _build_ffmpeg_stream(
        request=request,
        target_fps=settings.stream_target_fps,
        output_width=output_width,
        output_height=output_height,
    )
    if process.stdout is None:
        raise RuntimeError("ffmpeg stdout is not available")

    frame_size = output_width * output_height * 3
    frame_index = 0
    started = time.monotonic()
    track_states: dict[str, LiveTrackState] = {}
    local_to_global: dict[str, str] = {}
    next_track_index = 1
    recent_entries: deque[tuple[float, str]] = deque()
    suppression_zones: list[dict[str, float]] = []
    lifetime_track_counts: Counter[str] = Counter()
    use_opencv_window = False
    ffplay_sink: subprocess.Popen[bytes] | None = None
    if not args.headless:
        preferred_renderer = args.renderer
        if preferred_renderer in {"auto", "opencv"}:
            try:
                cv2.namedWindow("traffic_live_view", cv2.WINDOW_NORMAL)
                use_opencv_window = True
                LOGGER.info("viewer renderer | opencv")
            except cv2.error as exc:
                if preferred_renderer == "opencv":
                    raise
                LOGGER.warning("OpenCV GUI is unavailable, falling back to ffplay: %s", exc)
        if not use_opencv_window:
            ffplay_sink = _build_ffplay_sink(rendered_width, rendered_height, settings.stream_target_fps)
            if ffplay_sink.stdin is None:
                raise RuntimeError("ffplay stdin is not available")
            LOGGER.info("viewer renderer | ffplay")

    LOGGER.info(
        "viewer tuning | fps=%s imgsz=%s conf=%.2f overlap_iou=%.2f stale=%.1fs stitch_gap=%.1fs stitch_dist=%.1f min_hits=%s edge_hits=%s static_duration=%.1fs static_move=%.1f",
        settings.stream_target_fps,
        settings.yolo_imgsz,
        settings.yolo_confidence,
        tuning.overlap_iou_threshold,
        tuning.track_stale_seconds,
        tuning.stitch_gap_seconds,
        tuning.stitch_distance_px,
        tuning.min_track_hits,
        tuning.edge_min_track_hits,
        tuning.static_duration_seconds,
        tuning.static_movement_px,
    )

    try:
        while True:
            if args.max_seconds > 0 and (time.monotonic() - started) >= args.max_seconds:
                break
            raw = process.stdout.read(frame_size)
            if not raw or len(raw) < frame_size:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((output_height, output_width, 3))
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=list(COCO_TARGETS.keys()),
                conf=settings.yolo_confidence,
                imgsz=settings.yolo_imgsz,
                verbose=False,
                device="cpu",
            )
            frame_rows: list[dict[str, object]] = []
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                xyxy = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                track_ids = (
                    result.boxes.id.int().cpu().numpy().tolist()
                    if result.boxes.id is not None
                    else [None] * len(classes)
                )
                for box, conf, cls_id, track_id in zip(xyxy, confs, classes, track_ids):
                    raw_name = COCO_TARGETS.get(int(cls_id), str(int(cls_id)))
                    vehicle_class = map_vehicle_class(raw_name)
                    x1, y1, x2, y2 = map(float, box.tolist())
                    frame_rows.append(
                        {
                            "track_id": f"trk_{track_id}" if track_id is not None else None,
                            "vehicle_class": vehicle_class,
                            "confidence": float(conf),
                            "bbox_x1": x1,
                            "bbox_y1": y1,
                            "bbox_x2": x2,
                            "bbox_y2": y2,
                            "centroid_x": (x1 + x2) / 2.0,
                            "centroid_y": (y1 + y2) / 2.0,
                            "touches_edge": x1 <= 8 or x2 >= output_width - 8 or y1 <= 8 or y2 >= output_height - 8,
                        }
                    )

            frame_rows = _filter_rows(frame_rows, output_width, output_height, settings)
            frame_rows = _suppress_overlaps(frame_rows, overlap_iou_threshold=tuning.overlap_iou_threshold)
            frame_rows = _apply_suppression_zones(frame_rows, suppression_zones)
            before_counted = {
                state.global_track_id
                for state in track_states.values()
                if state.counted
            }
            now_ts = time.monotonic()
            next_track_index = _update_track_states(
                frame_rows,
                track_states,
                local_to_global,
                next_track_index,
                now_ts,
                tuning=tuning,
                suppression_zones=suppression_zones,
            )
            active_track_counts = _stable_track_counts(track_states)
            frame_class_counts = Counter(str(row["vehicle_class"]) for row in frame_rows)
            for state in track_states.values():
                if state.counted and state.global_track_id not in before_counted:
                    recent_entries.append((now_ts, state.stable_class))
                    lifetime_track_counts[state.stable_class] += 1
            cutoff = now_ts - 10.0
            while recent_entries and recent_entries[0][0] < cutoff:
                recent_entries.popleft()
            recent_entry_counts = Counter(vehicle_class for _, vehicle_class in recent_entries)
            rendered = _draw_frame(
                frame,
                frame_rows,
                frame_class_counts,
                active_track_counts,
                lifetime_track_counts,
                recent_entry_counts,
                Path(settings.yolo_model_path).name,
            )

            if frame_index % max(settings.stream_target_fps, 1) == 0:
                LOGGER.info(
                    "live frame | t=%ss | frame_counts=%s | active_tracks=%s | lifetime_tracks=%s | new_last_10s=%s | suppression_zones=%s",
                    int(time.monotonic() - started),
                    dict(frame_class_counts),
                    dict(active_track_counts),
                    dict(lifetime_track_counts),
                    dict(recent_entry_counts),
                    len(suppression_zones),
                )

            if use_opencv_window:
                cv2.imshow("traffic_live_view", rendered)
                key = cv2.waitKey(1) & 0xFF
                if key in {27, ord("q")}:
                    break
            elif ffplay_sink is not None:
                try:
                    ffplay_sink.stdin.write(rendered.tobytes())
                    ffplay_sink.stdin.flush()
                except (BrokenPipeError, OSError):
                    break
            frame_index += 1
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if ffplay_sink is not None:
            if ffplay_sink.stdin is not None:
                ffplay_sink.stdin.close()
            if ffplay_sink.poll() is None:
                ffplay_sink.kill()
                ffplay_sink.wait(timeout=5)
        if use_opencv_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
