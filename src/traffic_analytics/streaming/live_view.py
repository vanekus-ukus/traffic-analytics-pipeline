from __future__ import annotations

import argparse
from collections import Counter
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


def _draw_frame(
    frame: np.ndarray,
    rows: list[dict[str, object]],
    class_counts: Counter[str],
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

    stats = ", ".join(f"{name}={value}" for name, value in sorted(class_counts.items())) or "-"
    cv2.rectangle(output, (12, 12), (output.shape[1] - 12, 86), (25, 25, 25), -1)
    cv2.putText(output, f"model={model_name}", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(output, f"frame_counts: {stats}", (24, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return output


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
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    settings = get_settings()
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
    if not args.headless:
        cv2.namedWindow("traffic_live_view", cv2.WINDOW_NORMAL)

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
                for box, conf, cls_id in zip(xyxy, confs, classes):
                    raw_name = COCO_TARGETS.get(int(cls_id), str(int(cls_id)))
                    vehicle_class = map_vehicle_class(raw_name)
                    x1, y1, x2, y2 = map(float, box.tolist())
                    frame_rows.append(
                        {
                            "vehicle_class": vehicle_class,
                            "confidence": float(conf),
                            "bbox_x1": x1,
                            "bbox_y1": y1,
                            "bbox_x2": x2,
                            "bbox_y2": y2,
                        }
                    )

            frame_rows = _filter_rows(frame_rows, output_width, output_height, settings)
            class_counts = Counter(str(row["vehicle_class"]) for row in frame_rows)
            rendered = _draw_frame(frame, frame_rows, class_counts, Path(settings.yolo_model_path).name)

            if frame_index % max(settings.stream_target_fps, 1) == 0:
                LOGGER.info("live frame | t=%ss | frame_counts=%s", int(time.monotonic() - started), dict(class_counts))

            if not args.headless:
                cv2.imshow("traffic_live_view", rendered)
                key = cv2.waitKey(1) & 0xFF
                if key in {27, ord("q")}:
                    break
            frame_index += 1
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
