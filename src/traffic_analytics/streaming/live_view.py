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
PANEL_WIDTH = 420


@dataclass
class LiveTrackState:
    global_track_id: str
    local_track_id: str | None
    first_seen: float
    last_seen: float
    first_centroid_x: float
    first_centroid_y: float
    prev_centroid_x: float
    prev_centroid_y: float
    centroid_x: float
    centroid_y: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    touches_edge: bool
    speed_px_per_sec: float = 0.0
    speed_kmh_estimated: float = 0.0
    position_history: deque[tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=32))
    speed_samples: deque[float] = field(default_factory=lambda: deque(maxlen=12))
    class_votes: Counter[str] = field(default_factory=Counter)
    class_history: deque[tuple[str, float, float]] = field(default_factory=lambda: deque(maxlen=24))
    counted: bool = False
    counted_class: str | None = None
    suppressed: bool = False

    @property
    def stable_class(self) -> str:
        return _resolve_stable_class(self)


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
    stitch_min_iou: float = 0.02
    reid_memory_seconds: float = 12.0
    reid_distance_px: float = 180.0
    min_track_hits: int = 2
    edge_min_track_hits: int = 1
    person_min_track_hits: int = 3
    person_min_duration_seconds: float = 0.35
    static_duration_seconds: float = 6.0
    static_movement_px: float = 20.0
    static_min_hits: int = 10
    suppression_padding_px: float = 20.0
    speed_smoothing_alpha: float = 0.35
    speed_kmh_factor: float = 0.18
    speed_window_seconds: float = 2.0
    speed_max_jump_ratio: float = 2.4
    speed_reset_gap_seconds: float = 0.7
    speed_max_kmh_estimated: float = 160.0
    ghost_zone_min_hits: int = 6
    ghost_zone_window_seconds: float = 20.0
    ghost_zone_radius_px: float = 45.0


def _class_family(vehicle_class: str) -> str:
    if vehicle_class in {"car", "truck", "bus", "motorcycle"}:
        return "vehicle"
    return vehicle_class


def _should_upgrade_counted_class(old_class: str | None, new_class: str) -> bool:
    if not old_class or old_class == new_class:
        return False
    rank = {
        "car": 1,
        "truck": 2,
        "bus": 3,
    }
    if old_class in rank and new_class in rank:
        return rank[new_class] > rank[old_class]
    return False


def _reclassify_counted_track(
    state: LiveTrackState,
    lifetime_track_counts: Counter[str],
) -> None:
    if not state.counted:
        return
    if not _should_upgrade_counted_class(state.counted_class, state.stable_class):
        return
    previous_class = state.counted_class
    if previous_class and lifetime_track_counts.get(previous_class, 0) > 0:
        lifetime_track_counts[previous_class] -= 1
        if lifetime_track_counts[previous_class] <= 0:
            del lifetime_track_counts[previous_class]
    state.counted_class = state.stable_class
    lifetime_track_counts[state.counted_class] += 1


def _resolve_stable_class(state: LiveTrackState) -> str:
    if not state.class_history:
        return state.class_votes.most_common(1)[0][0] if state.class_votes else "unknown"

    weighted_scores: Counter[str] = Counter()
    for idx, (vehicle_class, confidence, area_ratio) in enumerate(state.class_history, start=1):
        recency_weight = 0.55 + (idx / len(state.class_history))
        size_weight = 1.0 + min(area_ratio * 45.0, 2.5)
        confidence_weight = max(0.35, confidence)
        weighted_scores[vehicle_class] += recency_weight * size_weight * confidence_weight

    current_class = weighted_scores.most_common(1)[0][0]
    vehicle_scores = {
        cls: weighted_scores.get(cls, 0.0)
        for cls in ("car", "truck", "bus", "motorcycle")
        if weighted_scores.get(cls, 0.0) > 0
    }
    if vehicle_scores:
        dominant_vehicle_class = max(vehicle_scores, key=vehicle_scores.get)
        dominant_score = vehicle_scores[dominant_vehicle_class]
        car_score = vehicle_scores.get("car", 0.0)
        if dominant_vehicle_class in {"bus", "truck"} and dominant_score >= max(car_score * 0.85, 2.0):
            return dominant_vehicle_class
        if dominant_vehicle_class == "motorcycle" and dominant_score >= max(car_score * 1.05, 1.5):
            return dominant_vehicle_class
    return current_class


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


def _state_row_iou(state: LiveTrackState, row: dict[str, object]) -> float:
    state_box = {
        "bbox_x1": state.bbox_x1,
        "bbox_y1": state.bbox_y1,
        "bbox_x2": state.bbox_x2,
        "bbox_y2": state.bbox_y2,
    }
    return _iou(state_box, row)


def _suppress_overlaps(rows: list[dict[str, object]], overlap_iou_threshold: float) -> list[dict[str, object]]:
    ordered = sorted(rows, key=lambda row: float(row["confidence"]), reverse=True)
    kept: list[dict[str, object]] = []
    for row in ordered:
        drop = False
        for kept_row in kept:
            same_class = str(row["vehicle_class"]) == str(kept_row["vehicle_class"])
            if same_class and _iou(row, kept_row) >= overlap_iou_threshold:
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


def _top_speed_track(track_states: dict[str, LiveTrackState]) -> LiveTrackState | None:
    candidates = [state for state in track_states.values() if state.counted and state.speed_px_per_sec > 0]
    if not candidates:
        return None
    return max(candidates, key=lambda state: state.speed_px_per_sec)


def _zone_distance(a: dict[str, float], x: float, y: float) -> float:
    return ((a["cx"] - x) ** 2 + (a["cy"] - y) ** 2) ** 0.5


def _register_false_hotspot(
    state: LiveTrackState,
    hotspot_candidates: list[dict[str, float | str]],
    suppression_zones: list[dict[str, float | str]],
    now_ts: float,
    tuning: ViewerTuning,
) -> None:
    if tuning.ghost_zone_min_hits <= 0:
        return
    hits = sum(state.class_votes.values())
    duration = state.last_seen - state.first_seen
    displacement = (
        (state.centroid_x - state.first_centroid_x) ** 2 + (state.centroid_y - state.first_centroid_y) ** 2
    ) ** 0.5
    target_family = "person" if state.stable_class == "person" else _class_family(state.stable_class)
    if target_family not in {"person", "vehicle"}:
        return
    if state.counted:
        return
    if hits > max(tuning.person_min_track_hits, 4):
        return
    if duration > max(tuning.person_min_duration_seconds, 0.8):
        return
    if displacement > tuning.static_movement_px:
        return

    cx = state.centroid_x
    cy = state.centroid_y
    matched = None
    for zone in hotspot_candidates:
        if zone["target_family"] != target_family:
            continue
        if _zone_distance(zone, cx, cy) <= tuning.ghost_zone_radius_px:
            matched = zone
            break
    if matched is None:
        hotspot_candidates.append(
            {
                "cx": cx,
                "cy": cy,
                "hits": 1.0,
                "first_seen": now_ts,
                "last_seen": now_ts,
                "x1": state.bbox_x1,
                "y1": state.bbox_y1,
                "x2": state.bbox_x2,
                "y2": state.bbox_y2,
                "target_family": target_family,
            }
        )
        return

    matched["cx"] = (float(matched["cx"]) * float(matched["hits"]) + cx) / (float(matched["hits"]) + 1.0)
    matched["cy"] = (float(matched["cy"]) * float(matched["hits"]) + cy) / (float(matched["hits"]) + 1.0)
    matched["hits"] = float(matched["hits"]) + 1.0
    matched["last_seen"] = now_ts
    matched["x1"] = min(float(matched["x1"]), state.bbox_x1)
    matched["y1"] = min(float(matched["y1"]), state.bbox_y1)
    matched["x2"] = max(float(matched["x2"]), state.bbox_x2)
    matched["y2"] = max(float(matched["y2"]), state.bbox_y2)

    if float(matched["hits"]) < tuning.ghost_zone_min_hits:
        return
    exists = any(
        zone.get("target_family") == target_family and _zone_distance(zone, float(matched["cx"]), float(matched["cy"])) <= tuning.ghost_zone_radius_px
        for zone in suppression_zones
    )
    if exists:
        return
    suppression_zones.append(
        {
            "x1": float(matched["x1"]) - tuning.suppression_padding_px,
            "y1": float(matched["y1"]) - tuning.suppression_padding_px,
            "x2": float(matched["x2"]) + tuning.suppression_padding_px,
            "y2": float(matched["y2"]) + tuning.suppression_padding_px,
            "cx": float(matched["cx"]),
            "cy": float(matched["cy"]),
            "target_family": target_family,
        }
    )


def _update_track_speed(state: LiveTrackState, now_ts: float, tuning: ViewerTuning) -> None:
    gap_seconds = now_ts - state.last_seen
    if gap_seconds > tuning.speed_reset_gap_seconds:
        state.position_history.clear()
        state.speed_samples.clear()
        state.speed_px_per_sec = 0.0
        state.speed_kmh_estimated = 0.0

    state.position_history.append((now_ts, state.centroid_x, state.centroid_y))
    cutoff = now_ts - tuning.speed_window_seconds
    while len(state.position_history) > 2 and state.position_history[0][0] < cutoff:
        state.position_history.popleft()

    instant_speed: float | None = None
    if now_ts > state.last_seen and gap_seconds <= tuning.speed_reset_gap_seconds:
        dt = gap_seconds
        if dt > 0:
            instant_speed = (
                ((state.centroid_x - state.prev_centroid_x) ** 2 + (state.centroid_y - state.prev_centroid_y) ** 2) ** 0.5
            ) / dt

    window_speed: float | None = None
    if len(state.position_history) >= 2:
        first_ts, first_x, first_y = state.position_history[0]
        last_ts, last_x, last_y = state.position_history[-1]
        window_dt = last_ts - first_ts
        if window_dt > 0:
            window_speed = (((last_x - first_x) ** 2 + (last_y - first_y) ** 2) ** 0.5) / window_dt

    candidates = [value for value in (instant_speed, window_speed) if value is not None and value >= 0]
    if not candidates:
        return

    raw_speed = float(np.median(candidates))
    if state.speed_samples:
        baseline = float(np.median(state.speed_samples))
        if baseline > 0 and raw_speed > baseline * tuning.speed_max_jump_ratio:
            raw_speed = baseline * tuning.speed_max_jump_ratio
    state.speed_samples.append(raw_speed)
    sample_median = float(np.median(state.speed_samples))

    alpha = min(max(tuning.speed_smoothing_alpha, 0.0), 1.0)
    if state.speed_px_per_sec <= 0:
        state.speed_px_per_sec = sample_median
    else:
        state.speed_px_per_sec = (alpha * sample_median) + ((1 - alpha) * state.speed_px_per_sec)
    state.speed_kmh_estimated = state.speed_px_per_sec * tuning.speed_kmh_factor
    state.speed_kmh_estimated = min(state.speed_kmh_estimated, tuning.speed_max_kmh_estimated)


def _should_count_track(state: LiveTrackState, hits: int, tuning: ViewerTuning) -> bool:
    stable_class = state.stable_class
    duration = state.last_seen - state.first_seen
    if stable_class == "person":
        required_hits = tuning.person_min_track_hits
        return hits >= required_hits and duration >= tuning.person_min_duration_seconds
    required_hits = tuning.edge_min_track_hits if state.touches_edge else tuning.min_track_hits
    return hits >= required_hits


def _row_center_in_zone(row: dict[str, object], zone: dict[str, float]) -> bool:
    cx = float(row["centroid_x"])
    cy = float(row["centroid_y"])
    return zone["x1"] <= cx <= zone["x2"] and zone["y1"] <= cy <= zone["y2"]


def _apply_suppression_zones(rows: list[dict[str, object]], suppression_zones: list[dict[str, float]]) -> list[dict[str, object]]:
    if not suppression_zones:
        return rows
    filtered: list[dict[str, object]] = []
    for row in rows:
        suppressed = False
        row_family = _class_family(str(row["vehicle_class"]))
        for zone in suppression_zones:
            target_family = str(zone.get("target_family", ""))
            if target_family and target_family != row_family:
                continue
            if _row_center_in_zone(row, zone):
                suppressed = True
                break
        if suppressed:
            continue
        filtered.append(row)
    return filtered


def _assign_global_track(
    row: dict[str, object],
    track_states: dict[str, LiveTrackState],
    retired_track_states: dict[str, LiveTrackState],
    local_to_global: dict[str, str],
    blocked_global_ids: set[str],
    now_ts: float,
    next_track_index: int,
    max_gap_seconds: float,
    max_distance_px: float,
    min_iou: float,
    reid_memory_seconds: float,
    reid_distance_px: float,
) -> tuple[str, int]:
    local_track_id = row.get("track_id")
    if local_track_id and local_track_id in local_to_global:
        mapped_id = local_to_global[local_track_id]
        if mapped_id not in blocked_global_ids:
            return mapped_id, next_track_index

    family = _class_family(str(row["vehicle_class"]))
    cx = float(row["centroid_x"])
    cy = float(row["centroid_y"])
    best_id: str | None = None
    best_score: tuple[float, float] | None = None
    for global_id, state in track_states.items():
        if global_id in blocked_global_ids:
            continue
        if now_ts - state.last_seen > max_gap_seconds:
            continue
        if _class_family(state.stable_class) != family:
            continue
        distance = ((state.centroid_x - cx) ** 2 + (state.centroid_y - cy) ** 2) ** 0.5
        if distance > max_distance_px:
            continue
        bbox_iou = _state_row_iou(state, row)
        if bbox_iou < min_iou and distance > max_distance_px * 0.55:
            continue
        score = (bbox_iou, -distance)
        if best_score is None or score > best_score:
            best_score = score
            best_id = global_id

    if best_id is None:
        for global_id, state in retired_track_states.items():
            if now_ts - state.last_seen > reid_memory_seconds:
                continue
            if _class_family(state.stable_class) != family:
                continue
            distance = ((state.centroid_x - cx) ** 2 + (state.centroid_y - cy) ** 2) ** 0.5
            if distance > reid_distance_px:
                continue
            bbox_iou = _state_row_iou(state, row)
            score = (bbox_iou, -distance)
            if best_score is None or score > best_score:
                best_score = score
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
    retired_track_states: dict[str, LiveTrackState],
    hotspot_candidates: list[dict[str, float | str]],
    local_to_global: dict[str, str],
    next_track_index: int,
    now_ts: float,
    tuning: ViewerTuning,
    suppression_zones: list[dict[str, float]],
) -> int:
    stale_ids = [track_id for track_id, state in track_states.items() if now_ts - state.last_seen > tuning.track_stale_seconds]
    for track_id in stale_ids:
        state = track_states.pop(track_id)
        if state.counted:
            retired_track_states[track_id] = state
        else:
            _register_false_hotspot(state, hotspot_candidates, suppression_zones, now_ts, tuning)

    expired_retired_ids = [
        track_id for track_id, state in retired_track_states.items() if now_ts - state.last_seen > tuning.reid_memory_seconds
    ]
    for track_id in expired_retired_ids:
        del retired_track_states[track_id]

    known_ids = set(track_states) | set(retired_track_states)
    stale_local = [local_id for local_id, global_id in local_to_global.items() if global_id not in known_ids]
    for local_id in stale_local:
        del local_to_global[local_id]

    assigned_global_ids: set[str] = set()
    for row in rows:
        global_id, next_track_index = _assign_global_track(
            row,
            track_states,
            retired_track_states,
            local_to_global,
            assigned_global_ids,
            now_ts,
            next_track_index,
            max_gap_seconds=tuning.stitch_gap_seconds,
            max_distance_px=tuning.stitch_distance_px,
            min_iou=tuning.stitch_min_iou,
            reid_memory_seconds=tuning.reid_memory_seconds,
            reid_distance_px=tuning.reid_distance_px,
        )
        assigned_global_ids.add(global_id)
        state = track_states.get(global_id)
        if state is None:
            state = retired_track_states.pop(global_id, None)
            if state is None:
                state = LiveTrackState(
                    global_track_id=global_id,
                    local_track_id=row.get("track_id"),
                    first_seen=now_ts,
                    last_seen=now_ts,
                    first_centroid_x=float(row["centroid_x"]),
                    first_centroid_y=float(row["centroid_y"]),
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
                state.position_history.append((now_ts, state.centroid_x, state.centroid_y))
            track_states[global_id] = state
        state.local_track_id = row.get("track_id")
        prev_seen = state.last_seen
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
        bbox_area_ratio = max(
            0.0,
            ((state.bbox_x2 - state.bbox_x1) * (state.bbox_y2 - state.bbox_y1))
            / max(float(row.get("frame_area", 1.0)), 1.0),
        )
        state.class_history.append((str(row["vehicle_class"]), float(row["confidence"]), bbox_area_ratio))
        if now_ts > prev_seen:
            _update_track_speed(state, now_ts, tuning)
        row["stable_track_id"] = global_id
        row["vehicle_class"] = state.stable_class

        hits = sum(state.class_votes.values())
        if not state.counted:
            if _should_count_track(state, hits, tuning):
                state.counted = True
                state.counted_class = state.stable_class
        duration = state.last_seen - state.first_seen
        total_displacement = (
            (state.centroid_x - state.first_centroid_x) ** 2 + (state.centroid_y - state.first_centroid_y) ** 2
        ) ** 0.5
        if (
            not state.suppressed
            and state.counted
            and _class_family(state.stable_class) == "vehicle"
            and duration >= tuning.static_duration_seconds
            and hits >= tuning.static_min_hits
            and total_displacement <= tuning.static_movement_px
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
    top_speed_state: LiveTrackState | None,
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

    canvas = np.zeros((output.shape[0], output.shape[1] + PANEL_WIDTH, 3), dtype=np.uint8)
    canvas[:, : output.shape[1], :] = output
    canvas[:, output.shape[1] :, :] = (25, 25, 25)

    x0 = output.shape[1] + 20
    cv2.putText(canvas, f"model={model_name}", (x0, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    classes = ["car", "truck", "bus", "motorcycle", "bicycle", "person"]
    table_top = 64
    row_h = 34
    col_class = x0
    col_frame = x0 + 130
    col_active = x0 + 205
    col_total = x0 + 290

    cv2.putText(canvas, "class", (col_class, table_top), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 220, 255), 2)
    cv2.putText(canvas, "frame", (col_frame, table_top), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 220, 255), 2)
    cv2.putText(canvas, "active", (col_active, table_top), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 220, 255), 2)
    cv2.putText(canvas, "total", (col_total, table_top), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 220, 255), 2)
    cv2.line(canvas, (x0, table_top + 10), (output.shape[1] + PANEL_WIDTH - 20, table_top + 10), (70, 70, 70), 1)

    for idx, cls in enumerate(classes, start=1):
        y = table_top + idx * row_h
        cv2.putText(canvas, cls, (col_class, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas, str(frame_class_counts.get(cls, 0)), (col_frame, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas, str(active_track_counts.get(cls, 0)), (col_active, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas, str(lifetime_track_counts.get(cls, 0)), (col_total, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    recent_top = table_top + (len(classes) + 2) * row_h
    cv2.putText(canvas, "new_last_10s", (x0, recent_top), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (180, 220, 255), 2)
    for idx, cls in enumerate(classes, start=1):
        y = recent_top + idx * 28
        cv2.putText(canvas, f"{cls}: {recent_entry_counts.get(cls, 0)}", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    speed_top = recent_top + (len(classes) + 2) * 28
    cv2.putText(canvas, "top_speed", (x0, speed_top), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (180, 220, 255), 2)
    if top_speed_state is None:
        cv2.putText(canvas, "-", (x0, speed_top + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    else:
        cv2.putText(
            canvas,
            f"{top_speed_state.stable_class}  {top_speed_state.speed_px_per_sec:.1f} px/s",
            (x0, speed_top + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            canvas,
            f"~{top_speed_state.speed_kmh_estimated:.1f} km/h est",
            (x0, speed_top + 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (255, 255, 255),
            2,
        )
    return canvas


def _rendered_frame_size(frame_width: int, frame_height: int) -> tuple[int, int]:
    return frame_width + PANEL_WIDTH, frame_height


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
    parser.add_argument("--stitch-min-iou", type=float, default=None)
    parser.add_argument("--reid-memory-seconds", type=float, default=None)
    parser.add_argument("--reid-distance-px", type=float, default=None)
    parser.add_argument("--min-track-hits", type=int, default=None)
    parser.add_argument("--edge-min-track-hits", type=int, default=None)
    parser.add_argument("--person-min-track-hits", type=int, default=None)
    parser.add_argument("--person-min-duration-seconds", type=float, default=None)
    parser.add_argument("--static-duration-seconds", type=float, default=None)
    parser.add_argument("--static-movement-px", type=float, default=None)
    parser.add_argument("--static-min-hits", type=int, default=None)
    parser.add_argument("--suppression-padding-px", type=float, default=None)
    parser.add_argument("--speed-smoothing-alpha", type=float, default=None)
    parser.add_argument("--speed-kmh-factor", type=float, default=None)
    parser.add_argument("--speed-window-seconds", type=float, default=None)
    parser.add_argument("--speed-max-jump-ratio", type=float, default=None)
    parser.add_argument("--speed-reset-gap-seconds", type=float, default=None)
    parser.add_argument("--speed-max-kmh-estimated", type=float, default=None)
    parser.add_argument("--ghost-zone-min-hits", type=int, default=None)
    parser.add_argument("--ghost-zone-window-seconds", type=float, default=None)
    parser.add_argument("--ghost-zone-radius-px", type=float, default=None)
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
        tuning.stitch_distance_px = 160.0
        tuning.stitch_min_iou = 0.04
        tuning.reid_memory_seconds = 15.0
        tuning.reid_distance_px = 220.0
        tuning.min_track_hits = 1
        tuning.edge_min_track_hits = 1
        tuning.person_min_track_hits = 3
        tuning.person_min_duration_seconds = 0.35
        tuning.static_duration_seconds = 8.0
        tuning.static_movement_px = 24.0
        tuning.static_min_hits = 14
        tuning.overlap_iou_threshold = 0.60
        tuning.speed_reset_gap_seconds = 0.6
        tuning.speed_max_kmh_estimated = 140.0
        tuning.ghost_zone_min_hits = 0
        tuning.ghost_zone_window_seconds = 20.0
        tuning.ghost_zone_radius_px = 45.0
    if args.overlap_iou is not None:
        tuning.overlap_iou_threshold = args.overlap_iou
    if args.track_stale_seconds is not None:
        tuning.track_stale_seconds = args.track_stale_seconds
    if args.stitch_gap_seconds is not None:
        tuning.stitch_gap_seconds = args.stitch_gap_seconds
    if args.stitch_distance_px is not None:
        tuning.stitch_distance_px = args.stitch_distance_px
    if args.stitch_min_iou is not None:
        tuning.stitch_min_iou = args.stitch_min_iou
    if args.reid_memory_seconds is not None:
        tuning.reid_memory_seconds = args.reid_memory_seconds
    if args.reid_distance_px is not None:
        tuning.reid_distance_px = args.reid_distance_px
    if args.min_track_hits is not None:
        tuning.min_track_hits = args.min_track_hits
    if args.edge_min_track_hits is not None:
        tuning.edge_min_track_hits = args.edge_min_track_hits
    if args.person_min_track_hits is not None:
        tuning.person_min_track_hits = args.person_min_track_hits
    if args.person_min_duration_seconds is not None:
        tuning.person_min_duration_seconds = args.person_min_duration_seconds
    if args.static_duration_seconds is not None:
        tuning.static_duration_seconds = args.static_duration_seconds
    if args.static_movement_px is not None:
        tuning.static_movement_px = args.static_movement_px
    if args.static_min_hits is not None:
        tuning.static_min_hits = args.static_min_hits
    if args.suppression_padding_px is not None:
        tuning.suppression_padding_px = args.suppression_padding_px
    if args.speed_smoothing_alpha is not None:
        tuning.speed_smoothing_alpha = args.speed_smoothing_alpha
    if args.speed_kmh_factor is not None:
        tuning.speed_kmh_factor = args.speed_kmh_factor
    if args.speed_window_seconds is not None:
        tuning.speed_window_seconds = args.speed_window_seconds
    if args.speed_max_jump_ratio is not None:
        tuning.speed_max_jump_ratio = args.speed_max_jump_ratio
    if args.speed_reset_gap_seconds is not None:
        tuning.speed_reset_gap_seconds = args.speed_reset_gap_seconds
    if args.speed_max_kmh_estimated is not None:
        tuning.speed_max_kmh_estimated = args.speed_max_kmh_estimated
    if args.ghost_zone_min_hits is not None:
        tuning.ghost_zone_min_hits = args.ghost_zone_min_hits
    if args.ghost_zone_window_seconds is not None:
        tuning.ghost_zone_window_seconds = args.ghost_zone_window_seconds
    if args.ghost_zone_radius_px is not None:
        tuning.ghost_zone_radius_px = args.ghost_zone_radius_px
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
    retired_track_states: dict[str, LiveTrackState] = {}
    local_to_global: dict[str, str] = {}
    next_track_index = 1
    recent_entries: deque[tuple[float, str]] = deque()
    suppression_zones: list[dict[str, float]] = []
    ghost_zone_candidates: list[dict[str, float | str]] = []
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
        "viewer tuning | fps=%s imgsz=%s conf=%.2f overlap_iou=%.2f stale=%.1fs stitch_gap=%.1fs stitch_dist=%.1f stitch_min_iou=%.2f reid_memory=%.1fs reid_dist=%.1f min_hits=%s edge_hits=%s person_hits=%s person_duration=%.2fs static_duration=%.1fs static_move=%.1f static_min_hits=%s speed_alpha=%.2f speed_factor=%.3f speed_window=%.1fs speed_jump=%.2f speed_reset=%.2fs speed_cap=%.1f ghost_hits=%s ghost_window=%.1fs ghost_radius=%.1f",
        settings.stream_target_fps,
        settings.yolo_imgsz,
        settings.yolo_confidence,
        tuning.overlap_iou_threshold,
        tuning.track_stale_seconds,
        tuning.stitch_gap_seconds,
        tuning.stitch_distance_px,
        tuning.stitch_min_iou,
        tuning.reid_memory_seconds,
        tuning.reid_distance_px,
        tuning.min_track_hits,
        tuning.edge_min_track_hits,
        tuning.person_min_track_hits,
        tuning.person_min_duration_seconds,
        tuning.static_duration_seconds,
        tuning.static_movement_px,
        tuning.static_min_hits,
        tuning.speed_smoothing_alpha,
        tuning.speed_kmh_factor,
        tuning.speed_window_seconds,
        tuning.speed_max_jump_ratio,
        tuning.speed_reset_gap_seconds,
        tuning.speed_max_kmh_estimated,
        tuning.ghost_zone_min_hits,
        tuning.ghost_zone_window_seconds,
        tuning.ghost_zone_radius_px,
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
                            "frame_area": float(output_width * output_height),
                            "frame_height": float(output_height),
                        }
                    )

            frame_rows = _filter_rows(frame_rows, output_width, output_height, settings)
            frame_rows = _suppress_overlaps(frame_rows, overlap_iou_threshold=tuning.overlap_iou_threshold)
            frame_rows = _apply_suppression_zones(frame_rows, suppression_zones)
            before_counted = {
                state.global_track_id
                for state in list(track_states.values()) + list(retired_track_states.values())
                if state.counted
            }
            now_ts = time.monotonic()
            next_track_index = _update_track_states(
                frame_rows,
                track_states,
                retired_track_states,
                ghost_zone_candidates,
                local_to_global,
                next_track_index,
                now_ts,
                tuning=tuning,
                suppression_zones=suppression_zones,
            )
            active_track_counts = _stable_track_counts(track_states)
            frame_class_counts = Counter(str(row["vehicle_class"]) for row in frame_rows)
            for state in track_states.values():
                _reclassify_counted_track(state, lifetime_track_counts)
                if state.counted and state.global_track_id not in before_counted:
                    counted_class = state.counted_class or state.stable_class
                    recent_entries.append((now_ts, counted_class))
                    lifetime_track_counts[counted_class] += 1
            cutoff = now_ts - 10.0
            while recent_entries and recent_entries[0][0] < cutoff:
                recent_entries.popleft()
            recent_entry_counts = Counter(vehicle_class for _, vehicle_class in recent_entries)
            top_speed_state = _top_speed_track(track_states)
            rendered = _draw_frame(
                frame,
                frame_rows,
                frame_class_counts,
                active_track_counts,
                lifetime_track_counts,
                recent_entry_counts,
                top_speed_state,
                Path(settings.yolo_model_path).name,
            )

            if frame_index % max(settings.stream_target_fps, 1) == 0:
                top_speed_text = (
                    f"{top_speed_state.stable_class}:{top_speed_state.speed_px_per_sec:.1f}px/s ~{top_speed_state.speed_kmh_estimated:.1f}km/h"
                    if top_speed_state is not None
                    else "-"
                )
                LOGGER.info(
                    "live frame | t=%ss | frame_counts=%s | active_tracks=%s | lifetime_tracks=%s | new_last_10s=%s | top_speed=%s | suppression_zones=%s",
                    int(time.monotonic() - started),
                    dict(frame_class_counts),
                    dict(active_track_counts),
                    dict(lifetime_track_counts),
                    dict(recent_entry_counts),
                    top_speed_text,
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
