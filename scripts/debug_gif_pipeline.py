from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

from traffic_analytics.common.vehicle_mapping import map_vehicle_class
from traffic_analytics.config.settings import get_settings
from traffic_analytics.streaming.yolo_backend import (
    COCO_TARGETS,
    VEHICLE_CLASSES,
    _motion_overlap_ratio,
    build_motion_mask,
    filter_low_quality_tracks,
    filter_static_tracks,
)


def _iter_frames(video_path: str, frame_step: int):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {video_path}")
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % frame_step == 0:
            yield frame_index, frame
        frame_index += 1
    capture.release()


def _draw_mask(frame: np.ndarray, motion_mask: np.ndarray | None) -> np.ndarray:
    output = frame.copy()
    if motion_mask is None:
        return output
    overlay = np.zeros_like(output)
    overlay[:, :, 0] = np.where(motion_mask > 0, 255, 0)
    overlay[:, :, 1] = np.where(motion_mask > 0, 180, 0)
    return cv2.addWeighted(output, 0.88, overlay, 0.12, 0)


def _draw_boxes(frame: np.ndarray, rows: pd.DataFrame, motion_mask: np.ndarray | None) -> np.ndarray:
    output = _draw_mask(frame, motion_mask)
    for _, row in rows.iterrows():
        x1, y1, x2, y2 = map(int, [row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]])
        color = tuple(int(v) for v in row["draw_color"])
        label = row["draw_label"]
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
    return output


def _save_gif(frame_paths: list[Path], gif_path: Path, duration_ms: int) -> None:
    images = [Image.open(path).convert("P", palette=Image.ADAPTIVE) for path in frame_paths]
    if not images:
        return
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def _base_record(
    frame_index: int,
    track_id: str,
    raw_name: str,
    conf: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_width: int,
    frame_height: int,
    motion_ratio: float,
    touches_edge: bool,
) -> dict[str, object]:
    vehicle_class = map_vehicle_class(raw_name)
    bbox_w = max(1.0, x2 - x1)
    bbox_h = max(1.0, y2 - y1)
    return {
        "frame_no": frame_index,
        "track_id": track_id,
        "class_name_raw": raw_name,
        "vehicle_class": vehicle_class,
        "confidence": float(conf),
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2,
        "centroid_x": (x1 + x2) / 2,
        "centroid_y": (y1 + y2) / 2,
        "motion_ratio": motion_ratio,
        "bbox_area_ratio": max(0.0, ((x2 - x1) * (y2 - y1)) / max(frame_width * frame_height, 1)),
        "bbox_aspect_ratio": bbox_w / bbox_h,
        "bbox_bottom_ratio": y2 / max(frame_height, 1),
        "touches_edge": touches_edge,
    }


def _run_stage(
    video_path: str,
    stage_name: str,
    settings,
    model: YOLO,
    motion_mask: np.ndarray | None,
    output_dir: Path,
    yolo_confidence: float,
    yolo_imgsz: int,
    filter_motion: bool,
    filter_geometry: bool,
    filter_static: bool,
    filter_quality: bool,
) -> dict[str, int]:
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    capture.release()
    frame_step = max(1, round(fps / max(settings.stream_target_fps, 1)))

    raw_records: list[dict[str, object]] = []
    frame_cache: dict[int, np.ndarray] = {}

    for frame_index, frame in _iter_frames(video_path, frame_step):
        frame_cache[frame_index] = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        edge_margin_x = frame_width * settings.edge_border_ratio
        edge_margin_y = frame_height * settings.edge_border_ratio
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=list(COCO_TARGETS.keys()),
            conf=yolo_confidence,
            imgsz=yolo_imgsz,
            verbose=False,
            device="cpu",
        )
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue
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
            x1, y1, x2, y2 = map(float, box.tolist())
            motion_ratio = _motion_overlap_ratio(motion_mask, x1, y1, x2, y2)
            touches_edge = (
                x1 <= edge_margin_x
                or x2 >= frame_width - edge_margin_x
                or y1 <= edge_margin_y
                or y2 >= frame_height - edge_margin_y
            )
            raw_records.append(
                _base_record(
                    frame_index=frame_index,
                    track_id=f"trk_{track_id}" if track_id is not None else f"frame_{frame_index}_{len(raw_records)}",
                    raw_name=raw_name,
                    conf=float(conf),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    motion_ratio=motion_ratio,
                    touches_edge=touches_edge,
                )
            )

    events = pd.DataFrame.from_records(raw_records)
    if events.empty:
        return {"frames": 0, "detections": 0}

    if filter_motion:
        motion_min = settings.motion_bbox_min_ratio
        relaxed = motion_min * settings.edge_motion_relax_factor
        events = events.loc[
            events.apply(
                lambda row: row["motion_ratio"] >= (relaxed if bool(row["touches_edge"]) else motion_min),
                axis=1,
            )
        ].copy()
    if filter_geometry:
        events = events.loc[
            (events["bbox_area_ratio"] > 0)
            & (events["bbox_area_ratio"] <= settings.max_bbox_area_ratio)
        ].copy()
        vehicle_mask = events["vehicle_class"].isin(VEHICLE_CLASSES)
        keep_vehicle = (
            (events["bbox_aspect_ratio"] >= settings.vehicle_min_aspect_ratio)
            & (events["bbox_aspect_ratio"] <= settings.vehicle_max_aspect_ratio)
            & ~(
                (events["bbox_bottom_ratio"] < settings.upper_zone_vehicle_bottom_ratio)
                & (events["motion_ratio"] < settings.upper_zone_motion_ratio)
            )
        )
        events = events.loc[(~vehicle_mask) | keep_vehicle].copy()
    if filter_static:
        events["frame_ts"] = pd.to_datetime(events["frame_no"] / max(fps, 1.0), unit="s", utc=True)
        events = events.sort_values(["track_id", "frame_ts"]).reset_index(drop=True)
        events = filter_static_tracks(events, settings)
    else:
        events["frame_ts"] = pd.to_datetime(events["frame_no"] / max(fps, 1.0), unit="s", utc=True)
        events = events.sort_values(["track_id", "frame_ts"]).reset_index(drop=True)
    if filter_quality:
        events = filter_low_quality_tracks(events, settings)

    stage_dir = output_dir / stage_name
    frame_dir = stage_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    unique_frames = sorted(frame_cache.keys())
    saved_paths: list[Path] = []
    for seq_idx, frame_index in enumerate(unique_frames, start=1):
        frame = frame_cache[frame_index]
        rows = events.loc[events["frame_no"] == frame_index].copy()
        if stage_name.startswith("01_raw"):
            rows["inside_motion"] = rows["motion_ratio"] >= settings.motion_bbox_min_ratio
            rows["draw_color"] = rows["inside_motion"].map(lambda v: (0, 220, 0) if v else (0, 0, 255))
            rows["draw_label"] = rows.apply(
                lambda row: f"{row['vehicle_class']} {float(row['confidence']):.2f} {'IN' if row['inside_motion'] else 'OUT'}",
                axis=1,
            )
        else:
            rows["draw_color"] = [(0, 220, 0)] * len(rows)
            rows["draw_label"] = rows.apply(
                lambda row: f"{row['vehicle_class']} {float(row['confidence']):.2f}",
                axis=1,
            )
        plotted = _draw_boxes(frame, rows, motion_mask)
        frame_path = frame_dir / f"frame_{seq_idx:03d}.jpg"
        cv2.imwrite(str(frame_path), plotted)
        saved_paths.append(frame_path)

    _save_gif(saved_paths, stage_dir / f"{stage_name}.gif", duration_ms=int(1000 / max(settings.stream_target_fps, 1)))
    events[["frame_no", "track_id", "vehicle_class", "confidence", "motion_ratio", "touches_edge"]].to_csv(
        stage_dir / "detections.csv", index=False
    )
    return {"frames": len(saved_paths), "detections": int(len(events))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--raw-high-recall", action="store_true")
    parser.add_argument("--stages", default="")
    args = parser.parse_args()

    settings = get_settings()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    motion_mask = build_motion_mask(
        source_path=args.video,
        frame_step=max(1, round(25 / max(settings.stream_target_fps, 1))),
        max_frames=None,
        threshold=settings.motion_mask_threshold,
        debug_dir=output_dir / "motion_debug",
        roi_x_min_ratio=settings.motion_roi_x_min_ratio,
        roi_x_max_ratio=settings.motion_roi_x_max_ratio,
        roi_y_min_ratio=settings.motion_roi_y_min_ratio,
        roi_y_max_ratio=settings.motion_roi_y_max_ratio,
    )

    stages = [
        ("01_raw_current", settings.yolo_confidence, settings.yolo_imgsz, False, False, False, False),
        ("02_road_stage", settings.yolo_confidence, settings.yolo_imgsz, True, True, False, False),
        ("03_no_static_filter", settings.yolo_confidence, settings.yolo_imgsz, True, True, False, True),
        ("04_current_filter", settings.yolo_confidence, settings.yolo_imgsz, True, True, True, True),
    ]
    if args.raw_high_recall:
        stages.insert(1, ("01b_raw_high_recall", 0.15, 960, False, False, False, False))
    if args.stages:
        wanted = {item.strip() for item in args.stages.split(",") if item.strip()}
        stages = [stage for stage in stages if stage[0] in wanted]

    summary_rows = []
    for stage in stages:
        model = YOLO(str(settings.yolo_model_path))
        stats = _run_stage(
            video_path=args.video,
            stage_name=stage[0],
            settings=settings,
            model=model,
            motion_mask=motion_mask,
            output_dir=output_dir,
            yolo_confidence=stage[1],
            yolo_imgsz=stage[2],
            filter_motion=stage[3],
            filter_geometry=stage[4],
            filter_static=stage[5],
            filter_quality=stage[6],
        )
        summary_rows.append({"stage": stage[0], **stats})

    pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)


if __name__ == "__main__":
    main()
