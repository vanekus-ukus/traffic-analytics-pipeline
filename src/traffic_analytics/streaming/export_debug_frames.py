from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import cv2
import pandas as pd

from traffic_analytics.common.logging_utils import configure_logging
from traffic_analytics.config.settings import get_settings
from traffic_analytics.streaming.video_source import resolve_video_source
from traffic_analytics.streaming.yolo_backend import detect_and_track_video


def draw_annotations(frame, frame_events: pd.DataFrame) -> None:
    for _, row in frame_events.iterrows():
        x1, y1, x2, y2 = map(int, [row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]])
        label = f"{row['vehicle_class']} {float(row['confidence']):.2f}"
        color = (0, 220, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - text_h - 8)), (x1 + text_w + 6, y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 3, max(12, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export annotated screenshots from the current generic detector pipeline.")
    parser.add_argument("--source", default=None, help="Local file path or remote page/media URL.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--step-seconds", type=int, default=5)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    settings = get_settings()
    if args.source:
        settings.stream_source = args.source
    resolution = resolve_video_source(settings)
    source_path = (
        resolution.resolved_uri
        if resolution.source_type in {"remote_direct", "remote_page_direct"}
        else str(resolution.local_path)
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else settings.artifacts_dir / "detector_debug" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    events, _ = detect_and_track_video(
        settings=settings,
        source_path=source_path,
        fps=resolution.fps,
        width=resolution.width,
        height=resolution.height,
        artifacts_dir=output_dir / "_tmp_samples",
    )
    if events.empty:
        raise RuntimeError("Detector returned zero filtered events for this source.")

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source for frame export: {source_path}")

    fps = resolution.fps or cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_step = max(1, int(args.step_seconds * fps))
    eventful_frames = sorted(set(int(frame_no) for frame_no in events["frame_no"].dropna().astype(int)))
    selected_frames: list[int] = []
    cursor = eventful_frames[0]
    for frame_no in eventful_frames:
        if frame_no >= cursor:
            selected_frames.append(frame_no)
            cursor = frame_no + frame_step
        if len(selected_frames) >= args.samples:
            break
    if not selected_frames:
        selected_frames = eventful_frames[: args.samples]

    manifest: list[dict[str, object]] = []
    for exported, frame_no in enumerate(selected_frames, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if not ok:
            continue
        frame_events = events.loc[events["frame_no"].astype(int) == frame_no].copy()
        if frame_events.empty:
            continue
        draw_annotations(frame, frame_events)
        out_path = output_dir / f"detector_frame_{exported:03d}.jpg"
        cv2.imwrite(str(out_path), frame)
        manifest.append(
            {
                "image_path": str(out_path),
                "frame_no": int(frame_no),
                "timestamp_sec": round(frame_no / fps, 3),
                "source_type": resolution.source_type,
                "detections": [
                    {
                        "track_id": row["track_id"],
                        "class_name_raw": row["class_name_raw"],
                        "vehicle_class": row["vehicle_class"],
                        "confidence": round(float(row["confidence"]), 4),
                        "bbox": [
                            round(float(row["bbox_x1"]), 2),
                            round(float(row["bbox_y1"]), 2),
                            round(float(row["bbox_x2"]), 2),
                            round(float(row["bbox_y2"]), 2),
                        ],
                    }
                    for _, row in frame_events.iterrows()
                ],
            }
        )

    cap.release()
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
