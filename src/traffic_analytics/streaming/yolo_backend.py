from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from traffic_analytics.common.vehicle_mapping import map_vehicle_class
from traffic_analytics.config.settings import Settings
from traffic_analytics.training.scene_profile import build_scene_profile, save_scene_profile

LOGGER = logging.getLogger(__name__)

COCO_TARGETS = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}


def _frame_timestamp(base_time: datetime, frame_index: int, fps: float) -> datetime:
    return base_time + timedelta(seconds=frame_index / max(fps, 1.0))


def _iter_frames(video_path: str, frame_step: int, max_frames: int | None):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {video_path}")
    frame_index = 0
    yielded = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % frame_step == 0:
            yield frame_index, frame
            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                break
        frame_index += 1
    capture.release()


def build_motion_mask(
    source_path: str,
    frame_step: int,
    max_frames: int | None,
    threshold: float,
    debug_dir: Path,
    roi_x_min_ratio: float,
    roi_x_max_ratio: float,
    roi_y_min_ratio: float,
    roi_y_max_ratio: float,
) -> np.ndarray | None:
    subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)
    accumulator: np.ndarray | None = None
    sampled = 0
    original_shape: tuple[int, int] | None = None
    first_frame: np.ndarray | None = None

    for _, frame in _iter_frames(source_path, frame_step, max_frames):
        original_shape = frame.shape[:2]
        if first_frame is None:
            first_frame = frame.copy()
        small = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
        fg = subtractor.apply(small)
        fg = cv2.GaussianBlur(fg, (5, 5), 0)
        _, fg = cv2.threshold(fg, 200, 1, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        fg = cv2.dilate(fg, kernel, iterations=2)
        accumulator = fg.astype(np.float32) if accumulator is None else accumulator + fg
        sampled += 1

    if accumulator is None or sampled == 0 or original_shape is None:
        return None

    freq = accumulator / sampled
    mask_small = (freq >= threshold).astype(np.uint8) * 255
    kernel = np.ones((9, 9), np.uint8)
    mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, kernel)
    mask_small = cv2.dilate(mask_small, kernel, iterations=1)
    mask = cv2.resize(mask_small, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
    h, w = mask.shape[:2]
    x_min = max(0, min(w, int(w * roi_x_min_ratio)))
    x_max = max(x_min + 1, min(w, int(w * roi_x_max_ratio)))
    y_min = max(0, min(h, int(h * roi_y_min_ratio)))
    y_max = max(y_min + 1, min(h, int(h * roi_y_max_ratio)))
    clipped = np.zeros_like(mask)
    clipped[y_min:y_max, x_min:x_max] = mask[y_min:y_max, x_min:x_max]
    mask = clipped
    coverage = float((mask > 0).mean())
    if coverage <= 0.005 or coverage >= 0.85:
        LOGGER.info("Motion mask coverage %.4f outside useful range; disabling motion ROI filter", coverage)
        return None

    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / "motion_mask.png"), mask)
    if first_frame is not None:
        overlay = first_frame.copy()
        tint = np.zeros_like(overlay)
        tint[:, :, 0] = np.where(mask > 0, 255, 0)
        tint[:, :, 1] = np.where(mask > 0, 180, 0)
        overlay = cv2.addWeighted(overlay, 0.86, tint, 0.14, 0)
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
        cv2.imwrite(str(debug_dir / "motion_mask_overlay.jpg"), overlay)
    return mask


def filter_static_tracks(events: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    if events.empty:
        return events
    grouped = events.groupby("track_id")
    summary = grouped.agg(
        first_ts=("frame_ts", "min"),
        last_ts=("frame_ts", "max"),
        start_x=("centroid_x", "first"),
        start_y=("centroid_y", "first"),
        end_x=("centroid_x", "last"),
        end_y=("centroid_y", "last"),
    ).reset_index()
    summary["duration_sec"] = (summary["last_ts"] - summary["first_ts"]).dt.total_seconds()
    summary["distance_px"] = (
        (summary["end_x"] - summary["start_x"]).pow(2) + (summary["end_y"] - summary["start_y"]).pow(2)
    ) ** 0.5
    static_ids = set(
        summary.loc[
            (summary["duration_sec"] >= settings.static_track_min_duration_sec)
            & (summary["distance_px"] <= settings.static_track_max_distance_px),
            "track_id",
        ]
    )
    if not static_ids:
        return events
    LOGGER.info("Filtered %s static tracks as likely false positives", len(static_ids))
    return events.loc[~events["track_id"].isin(static_ids)].copy()


def filter_low_quality_tracks(events: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    if events.empty:
        return events
    grouped = events.groupby("track_id")
    summary = grouped.agg(
        detections_count=("frame_no", "size"),
        first_ts=("frame_ts", "min"),
        last_ts=("frame_ts", "max"),
        start_x=("centroid_x", "first"),
        start_y=("centroid_y", "first"),
        end_x=("centroid_x", "last"),
        end_y=("centroid_y", "last"),
        mean_confidence=("confidence", "mean"),
        touches_edge=("touches_edge", "max"),
    ).reset_index()
    summary["distance_px"] = (
        (summary["end_x"] - summary["start_x"]).pow(2) + (summary["end_y"] - summary["start_y"]).pow(2)
    ) ** 0.5
    standard_keep = (
        (summary["detections_count"] >= settings.track_min_detections)
        & (summary["distance_px"] >= settings.track_min_distance_px)
        & (summary["mean_confidence"] >= settings.track_min_confidence)
    )
    edge_keep = (
        summary["touches_edge"].fillna(False).astype(bool)
        & (summary["detections_count"] >= settings.edge_track_min_detections)
        & (summary["distance_px"] >= settings.edge_track_min_distance_px)
        & (summary["mean_confidence"] >= settings.edge_track_min_confidence)
    )
    keep_ids = set(summary.loc[standard_keep | edge_keep, "track_id"])
    removed = len(summary) - len(keep_ids)
    if removed > 0:
        LOGGER.info("Filtered %s low-quality tracks using generic road heuristics", removed)
    return events.loc[events["track_id"].isin(keep_ids)].copy()


def _motion_overlap_ratio(mask: np.ndarray | None, x1: float, y1: float, x2: float, y2: float) -> float:
    if mask is None:
        return 1.0
    h, w = mask.shape[:2]
    xi1 = max(0, min(w - 1, int(x1)))
    yi1 = max(0, min(h - 1, int(y1)))
    xi2 = max(xi1 + 1, min(w, int(x2)))
    yi2 = max(yi1 + 1, min(h, int(y2)))
    patch = mask[yi1:yi2, xi1:xi2]
    if patch.size == 0:
        return 0.0
    return float((patch > 0).mean())


def _draw_filtered_annotations(
    frame: np.ndarray,
    frame_events: pd.DataFrame,
    motion_mask: np.ndarray | None = None,
) -> np.ndarray:
    output = frame.copy()
    if motion_mask is not None:
        overlay = np.zeros_like(output)
        overlay[:, :, 0] = np.where(motion_mask > 0, 255, 0)
        overlay[:, :, 1] = np.where(motion_mask > 0, 180, 0)
        output = cv2.addWeighted(output, 0.88, overlay, 0.12, 0)
    for _, row in frame_events.iterrows():
        x1, y1, x2, y2 = map(int, [row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]])
        label = f"{row['vehicle_class']} {float(row['confidence']):.2f}"
        color = (0, 220, 0)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(output, (x1, max(0, y1 - text_h - 8)), (x1 + text_w + 6, y1), color, -1)
        cv2.putText(
            output,
            label,
            (x1 + 3, max(12, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return output


def export_annotated_video(
    source_path: str,
    events: pd.DataFrame,
    output_path: Path,
    fps: float | None = None,
) -> Path:
    capture = cv2.VideoCapture(source_path)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source for rendering: {source_path}")

    source_fps = fps or float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(source_fps, 1.0),
        (frame_width, frame_height),
    )

    if events.empty:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
        capture.release()
        writer.release()
        return output_path

    max_frame_gap = max(1, int(round(source_fps / 2.0)))
    grouped = {
        int(frame_no): frame_df.copy()
        for frame_no, frame_df in events.groupby("frame_no", sort=True)
    }
    active_tracks: dict[str, tuple[pd.Series, int]] = {}
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_index in grouped:
            frame_events = grouped[frame_index]
            for _, row in frame_events.iterrows():
                active_tracks[str(row["track_id"])] = (row, frame_index)

        valid_rows: list[pd.Series] = []
        for track_id, (row, last_frame) in list(active_tracks.items()):
            if frame_index - last_frame > max_frame_gap:
                del active_tracks[track_id]
                continue
            valid_rows.append(row)

        if valid_rows:
            rendered = _draw_filtered_annotations(frame, pd.DataFrame(valid_rows), motion_mask=None)
        else:
            rendered = frame
        writer.write(rendered)
        frame_index += 1

    capture.release()
    writer.release()
    return output_path


def detect_and_track_video(
    settings: Settings,
    source_path: str,
    fps: float | None,
    width: int | None,
    height: int | None,
    artifacts_dir: Path,
    base_time: datetime | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    effective_fps = fps or 25.0
    frame_step = max(1, round(effective_fps / max(settings.stream_target_fps, 1)))
    max_frames = None
    if settings.stream_max_seconds > 0:
        max_frames = max(1, round((settings.stream_max_seconds * effective_fps) / frame_step))

    model = YOLO(str(settings.yolo_model_path))
    scene_profile = None
    if settings.use_scene_profile:
        scene_profile = build_scene_profile(settings.tracking_fallback, settings.stream_target_fps)
        save_scene_profile(scene_profile, settings.artifacts_dir / "calibration" / "historical_scene_profile.json")

    motion_mask = None
    if settings.motion_roi_enabled:
        motion_mask = build_motion_mask(
            source_path=source_path,
            frame_step=frame_step,
            max_frames=max_frames,
            threshold=settings.motion_mask_threshold,
            debug_dir=settings.artifacts_dir / "motion_debug",
            roi_x_min_ratio=settings.motion_roi_x_min_ratio,
            roi_x_max_ratio=settings.motion_roi_x_max_ratio,
            roi_y_min_ratio=settings.motion_roi_y_min_ratio,
            roi_y_max_ratio=settings.motion_roi_y_max_ratio,
        )

    base_time = base_time or datetime.now(timezone.utc)
    records: list[dict[str, object]] = []
    sample_paths: list[str] = []
    sample_frames_cache: list[tuple[int, np.ndarray]] = []
    class_totals: dict[str, int] = {}
    seen_track_ids: set[str] = set()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for processed_idx, (frame_index, frame) in enumerate(_iter_frames(source_path, frame_step, max_frames)):
        frame_height, frame_width = frame.shape[:2]
        edge_margin_x = frame_width * settings.edge_border_ratio
        edge_margin_y = frame_height * settings.edge_border_ratio
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
        result = results[0]
        frame_ts = _frame_timestamp(base_time, frame_index, effective_fps)
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            ids = (
                result.boxes.id.int().cpu().numpy().tolist()
                if result.boxes.id is not None
                else [None] * len(classes)
            )
            for box, conf, cls_id, track_id in zip(xyxy, confs, classes, ids):
                raw_name = COCO_TARGETS.get(int(cls_id), str(int(cls_id)))
                vehicle_class = map_vehicle_class(raw_name)
                if scene_profile is not None:
                    vehicle_class = scene_profile.class_remap_for_inference.get(vehicle_class, vehicle_class)
                x1, y1, x2, y2 = map(float, box.tolist())
                bbox_w = max(1.0, x2 - x1)
                bbox_h = max(1.0, y2 - y1)
                bbox_aspect_ratio = bbox_w / bbox_h
                bbox_bottom_ratio = y2 / max(frame_height, 1)
                bbox_area_ratio = max(0.0, ((x2 - x1) * (y2 - y1)) / max(frame_width * frame_height, 1))
                touches_edge = (
                    x1 <= edge_margin_x
                    or x2 >= frame_width - edge_margin_x
                    or y1 <= edge_margin_y
                    or y2 >= frame_height - edge_margin_y
                )
                if bbox_area_ratio <= 0 or bbox_area_ratio > settings.max_bbox_area_ratio:
                    continue
                motion_ratio = _motion_overlap_ratio(motion_mask, x1, y1, x2, y2)
                effective_motion_min_ratio = settings.motion_bbox_min_ratio
                if touches_edge:
                    effective_motion_min_ratio *= settings.edge_motion_relax_factor
                if motion_ratio < effective_motion_min_ratio:
                    continue
                if vehicle_class in VEHICLE_CLASSES:
                    if (
                        bbox_aspect_ratio < settings.vehicle_min_aspect_ratio
                        or bbox_aspect_ratio > settings.vehicle_max_aspect_ratio
                    ):
                        continue
                    if (
                        bbox_bottom_ratio < settings.upper_zone_vehicle_bottom_ratio
                        and motion_ratio < settings.upper_zone_motion_ratio
                    ):
                        continue
                records.append(
                    {
                        "frame_ts": frame_ts,
                        "frame_no": frame_index,
                        "track_id": f"trk_{track_id}" if track_id is not None else f"frame_{frame_index}_{len(records)}",
                        "class_name_raw": raw_name,
                        "vehicle_class": vehicle_class,
                        "confidence": float(conf),
                        "centroid_x": (x1 + x2) / 2,
                        "centroid_y": (y1 + y2) / 2,
                        "bbox_x1": x1,
                        "bbox_y1": y1,
                        "bbox_x2": x2,
                        "bbox_y2": y2,
                        "motion_ratio": motion_ratio,
                        "bbox_area_ratio": bbox_area_ratio,
                        "bbox_aspect_ratio": bbox_aspect_ratio,
                        "touches_edge": touches_edge,
                    }
                )
                class_totals[vehicle_class] = class_totals.get(vehicle_class, 0) + 1
                if track_id is not None:
                    seen_track_ids.add(f"trk_{track_id}")
        if len(sample_frames_cache) < 20 and result.boxes is not None and len(result.boxes) > 0:
            sample_frames_cache.append((frame_index, frame.copy()))
        if progress_callback is not None:
            progress_callback(
                {
                    "frame_index": frame_index,
                    "processed_frames": processed_idx + 1,
                    "source_seconds": frame_index / max(effective_fps, 1.0),
                    "accepted_events": len(records),
                    "unique_tracks_seen": len(seen_track_ids),
                    "class_totals": dict(sorted(class_totals.items())),
                }
            )

    events = pd.DataFrame.from_records(records)
    if events.empty:
        events["speed_proxy"] = pd.Series(dtype=float)
        return events, sample_paths

    events = events.sort_values(["track_id", "frame_ts"]).reset_index(drop=True)
    events = filter_static_tracks(events, settings)
    events = filter_low_quality_tracks(events, settings)
    events = events.sort_values(["track_id", "frame_ts"]).reset_index(drop=True)
    grouped = events.groupby("track_id")
    events["prev_x"] = grouped["centroid_x"].shift(1)
    events["prev_y"] = grouped["centroid_y"].shift(1)
    events["prev_time"] = grouped["frame_ts"].shift(1)
    dt = (events["frame_ts"] - events["prev_time"]).dt.total_seconds()
    dx = events["centroid_x"] - events["prev_x"]
    dy = events["centroid_y"] - events["prev_y"]
    events["speed_proxy"] = ((dx.pow(2) + dy.pow(2)) ** 0.5 / dt).where(dt > 0)

    exported = 0
    for frame_index, frame in sample_frames_cache:
        frame_events = events.loc[events["frame_no"] == frame_index].copy()
        if frame_events.empty:
            continue
        plotted = _draw_filtered_annotations(frame, frame_events, motion_mask)
        sample_path = artifacts_dir / f"sample_frame_{exported + 1:03d}.jpg"
        cv2.imwrite(str(sample_path), plotted)
        sample_paths.append(str(sample_path))
        exported += 1
        if exported >= 5:
            break
    return events, sample_paths
