from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO
import ultralytics.data.dataset as yolo_dataset

from traffic_analytics.common.logging_utils import configure_logging
from traffic_analytics.config.settings import get_settings


class SequentialPool:
    def __init__(self, processes: int | None = None):
        self.processes = processes

    def __enter__(self) -> "SequentialPool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def imap(self, func, iterable):
        for item in iterable:
            yield func(item)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on the bootstrap dataset.")
    parser.add_argument("--data", default="datasets/yolo_bootstrap/data.yaml")
    parser.add_argument("--model", default=None, help="Base model. Defaults to configured yolov8n.pt.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", default="artifacts/train_runs")
    parser.add_argument("--name", default="traffic_bootstrap")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    settings = get_settings()
    model_path = Path(args.model) if args.model else settings.yolo_model_path
    project_dir = Path(args.project).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    yolo_dataset.ThreadPool = SequentialPool
    model = YOLO(str(model_path))
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz or settings.yolo_imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project_dir),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        workers=0,
        plots=False,
    )
    print(results.save_dir)


if __name__ == "__main__":
    main()
