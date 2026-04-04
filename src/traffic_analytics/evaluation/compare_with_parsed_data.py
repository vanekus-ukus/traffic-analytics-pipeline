from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from traffic_analytics.common.logging_utils import configure_logging
from traffic_analytics.config.settings import get_settings
from traffic_analytics.streaming.preannotated_backend import build_tracks, load_preannotated_events
from traffic_analytics.streaming.yolo_backend import detect_and_track_video


def build_tracks_from_yolo_events(events: pd.DataFrame) -> pd.DataFrame:
    tracks_source = events.rename(columns={"frame_ts": "detection_time"})
    tracks_source["x_cord_m"] = tracks_source["centroid_x"]
    tracks_source["y_cord_m"] = tracks_source["centroid_y"]
    tracks_source["direction"] = None
    return build_tracks(
        tracks_source[
            ["track_id", "vehicle_class", "detection_time", "x_cord_m", "y_cord_m", "direction"]
        ]
    )


def per_second_counts_from_events(
    events: pd.DataFrame,
    ts_col: str,
    track_col: str,
    class_col: str,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["second", "vehicle_class", "count"])
    df = events.copy()
    start_ts = df[ts_col].min()
    df["second"] = ((df[ts_col] - start_ts).dt.total_seconds()).astype(int)
    counts = (
        df.groupby(["second", class_col])[track_col]
        .nunique()
        .rename("count")
        .reset_index()
        .rename(columns={class_col: "vehicle_class"})
    )
    return counts


def build_comparison(gt_counts: pd.DataFrame, pred_counts: pd.DataFrame) -> pd.DataFrame:
    merged = gt_counts.merge(
        pred_counts,
        on=["second", "vehicle_class"],
        how="outer",
        suffixes=("_gt", "_pred"),
    ).fillna(0)
    merged["count_gt"] = merged["count_gt"].astype(int)
    merged["count_pred"] = merged["count_pred"].astype(int)
    merged["abs_error"] = (merged["count_gt"] - merged["count_pred"]).abs()
    return merged.sort_values(["vehicle_class", "second"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare YOLO output with parsed tracking data.")
    parser.add_argument("--output", default="artifacts/evaluation/comparison.json")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    settings = get_settings()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gt_events = load_preannotated_events(settings.tracking_fallback, fps=30.0)
    gt_tracks = build_tracks(gt_events)

    eval_settings = get_settings()
    eval_settings.stream_max_seconds = 84
    eval_settings.stream_target_fps = 5
    pred_events, sample_paths = detect_and_track_video(
        settings=eval_settings,
        source_path=str(settings.video_fallback),
        fps=30.0,
        width=1920,
        height=1080,
        artifacts_dir=settings.artifacts_dir / "evaluation_samples",
    )
    pred_tracks = build_tracks_from_yolo_events(pred_events)

    gt_track_counts = gt_tracks["vehicle_class"].value_counts().sort_index().to_dict()
    pred_track_counts = pred_tracks["vehicle_class"].value_counts().sort_index().to_dict()

    gt_counts = per_second_counts_from_events(gt_events, "detection_time", "track_id", "vehicle_class")
    pred_counts = per_second_counts_from_events(pred_events, "frame_ts", "track_id", "vehicle_class")
    comparison = build_comparison(gt_counts, pred_counts)
    class_mae = (
        comparison.groupby("vehicle_class")["abs_error"].mean().round(4).to_dict()
        if not comparison.empty
        else {}
    )

    report = {
        "ground_truth_unique_tracks": gt_track_counts,
        "predicted_unique_tracks": pred_track_counts,
        "class_mae_per_second": class_mae,
        "ground_truth_seconds": int(gt_counts["second"].max()) + 1 if not gt_counts.empty else 0,
        "predicted_seconds": int(pred_counts["second"].max()) + 1 if not pred_counts.empty else 0,
        "sample_paths": sample_paths,
        "comparison_preview": comparison.head(40).to_dict(orient="records"),
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
