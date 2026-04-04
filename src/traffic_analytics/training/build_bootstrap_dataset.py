from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

from traffic_analytics.common.logging_utils import configure_logging
from traffic_analytics.config.settings import get_settings
from traffic_analytics.training.scene_profile import build_scene_profile, save_scene_profile

COCO_TARGETS = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

YOLO_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(YOLO_CLASSES)}


def normalise_bbox(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[float, float, float, float]:
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return cx, cy, bw, bh


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a bootstrap YOLO dataset from video and historical priors.")
    parser.add_argument("--video", default=None, help="Video file to sample.")
    parser.add_argument("--output-dir", default="datasets/yolo_bootstrap")
    parser.add_argument("--sample-every-seconds", type=float, default=2.0)
    parser.add_argument("--max-seconds", type=int, default=120)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    settings = get_settings()
    video_path = Path(args.video) if args.video else settings.video_fallback
    output_dir = Path(args.output_dir)
    image_dirs = {
        "train": output_dir / "images" / "train",
        "val": output_dir / "images" / "val",
    }
    label_dirs = {
        "train": output_dir / "labels" / "train",
        "val": output_dir / "labels" / "val",
    }
    for directory in [*image_dirs.values(), *label_dirs.values()]:
        directory.mkdir(parents=True, exist_ok=True)

    profile = build_scene_profile(settings.tracking_fallback, settings.stream_target_fps)
    save_scene_profile(profile, output_dir / "scene_profile.json")

    model = YOLO(str(settings.yolo_model_path))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    frame_interval = max(1, round(fps * args.sample_every_seconds))
    max_frames = max(1, round(args.max_seconds * fps))
    val_every = max(2, round(1 / max(1e-6, 1 - args.train_ratio)))

    manifest: list[dict[str, object]] = []
    class_counter: Counter[str] = Counter()
    frame_index = 0
    sample_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame_index >= max_frames:
                break
            if frame_index % frame_interval != 0:
                frame_index += 1
                continue

            split = "val" if sample_index % val_every == 0 else "train"
            image_path = image_dirs[split] / f"frame_{sample_index:05d}.jpg"
            label_path = label_dirs[split] / f"frame_{sample_index:05d}.txt"

            results = model.predict(
                frame,
                classes=list(COCO_TARGETS.keys()),
                conf=settings.yolo_confidence,
                imgsz=settings.yolo_imgsz,
                verbose=False,
                device="cpu",
            )
            result = results[0]
            labels: list[str] = []
            detections_manifest: list[dict[str, object]] = []
            if result.boxes is not None and len(result.boxes) > 0:
                xyxy = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                for box, conf, cls_id in zip(xyxy, confs, classes):
                    raw_cls = COCO_TARGETS.get(int(cls_id), str(int(cls_id)))
                    calibrated_cls = profile.class_remap_for_inference.get(raw_cls, raw_cls)
                    if calibrated_cls not in CLASS_TO_IDX:
                        continue
                    x1, y1, x2, y2 = map(float, box.tolist())
                    cx, cy, bw, bh = normalise_bbox(x1, y1, x2, y2, width, height)
                    if bw <= 0 or bh <= 0:
                        continue
                    labels.append(f"{CLASS_TO_IDX[calibrated_cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    detections_manifest.append(
                        {
                            "raw_class": raw_cls,
                            "calibrated_class": calibrated_cls,
                            "confidence": float(conf),
                            "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                        }
                    )
                    class_counter[calibrated_cls] += 1
            if labels:
                cv2.imwrite(str(image_path), frame)
                label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
                manifest.append(
                    {
                        "image_path": str(image_path),
                        "label_path": str(label_path),
                        "split": split,
                        "frame_no": frame_index,
                        "timestamp_sec": round(frame_index / fps, 3),
                        "detections": detections_manifest,
                    }
                )
                sample_index += 1
            frame_index += 1
    finally:
        capture.release()

    (output_dir / "classes.txt").write_text("\n".join(YOLO_CLASSES) + "\n", encoding="utf-8")
    (output_dir / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                f"names: {json.dumps({i: name for i, name in enumerate(YOLO_CLASSES)}, ensure_ascii=False)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "video_path": str(video_path),
                "historical_profile": json.loads(profile.to_json()),
                "images_written": len(manifest),
                "class_distribution": dict(class_counter),
                "samples": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
