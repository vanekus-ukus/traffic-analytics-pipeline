from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from traffic_analytics.common.vehicle_mapping import map_vehicle_class
from traffic_analytics.streaming.preannotated_backend import build_tracks


@dataclass(slots=True)
class SceneProfile:
    source_csv: str
    observed_classes: list[str]
    class_frequencies: dict[str, int]
    class_remap_for_inference: dict[str, str]
    min_track_detections_10p: dict[str, int]
    min_track_duration_sec_10p: dict[str, float]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def build_scene_profile(tracking_csv: Path, stream_target_fps: int = 5) -> SceneProfile:
    df = pd.read_csv(tracking_csv, parse_dates=["detection_time"])
    df["vehicle_class"] = df["object_type"].map(map_vehicle_class)
    tracks = build_tracks(
        df.rename(columns={"track_id": "source_track_id"})[
            ["source_track_id", "vehicle_class", "detection_time", "x_cord_m", "y_cord_m", "direction"]
        ].rename(columns={"source_track_id": "track_id"})
    )

    class_frequencies = tracks["vehicle_class"].value_counts().sort_index().astype(int).to_dict()
    observed_classes = sorted(class_frequencies.keys())

    det_quantiles = (
        tracks.groupby("vehicle_class")["detections_count"].quantile(0.1).round().astype(int).to_dict()
        if not tracks.empty
        else {}
    )
    duration_quantiles = (
        tracks.groupby("vehicle_class")["duration_seconds"].quantile(0.1).round(2).to_dict()
        if not tracks.empty
        else {}
    )
    fps_scale = max(stream_target_fps, 1) / 30.0
    min_track_detections_10p = {
        cls: max(1, int(round(value * fps_scale))) for cls, value in det_quantiles.items()
    }

    class_remap_for_inference: dict[str, str] = {}
    if "car" in observed_classes:
        for raw_cls in ("truck", "bus"):
            if raw_cls not in observed_classes:
                class_remap_for_inference[raw_cls] = "car"

    return SceneProfile(
        source_csv=str(tracking_csv),
        observed_classes=observed_classes,
        class_frequencies=class_frequencies,
        class_remap_for_inference=class_remap_for_inference,
        min_track_detections_10p=min_track_detections_10p,
        min_track_duration_sec_10p=duration_quantiles,
    )


def save_scene_profile(profile: SceneProfile, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(profile.to_json(), encoding="utf-8")
