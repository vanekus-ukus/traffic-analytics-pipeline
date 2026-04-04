from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from traffic_analytics.common.vehicle_mapping import map_vehicle_class

LOGGER = logging.getLogger(__name__)


def load_preannotated_events(tracking_path: Path, fps: float | None) -> pd.DataFrame:
    df = pd.read_csv(tracking_path)
    for column in ["recording_at", "recording_end", "start_time", "end_time", "detection_time"]:
        df[column] = pd.to_datetime(df[column], errors="coerce", utc=True)
    df = df.sort_values(["track_id", "detection_time"]).reset_index(drop=True)
    df["vehicle_class"] = df["object_type"].map(map_vehicle_class)
    base_ts = df["recording_at"].dropna().min()
    if pd.isna(base_ts):
        base_ts = df["detection_time"].min()
    df["frame_offset_sec"] = (df["detection_time"] - base_ts).dt.total_seconds().clip(lower=0)
    effective_fps = fps or 30.0
    df["frame_no"] = (df["frame_offset_sec"] * effective_fps).round().astype(int)
    df["track_key"] = df["track_id"].astype(str)
    grouped = df.groupby("track_id")
    df["prev_x"] = grouped["x_cord_m"].shift(1)
    df["prev_y"] = grouped["y_cord_m"].shift(1)
    df["prev_time"] = grouped["detection_time"].shift(1)
    dx = (df["x_cord_m"] - df["prev_x"]).astype(float)
    dy = (df["y_cord_m"] - df["prev_y"]).astype(float)
    dt = (df["detection_time"] - df["prev_time"]).dt.total_seconds()
    df["speed_proxy"] = np.where(
        (dt > 0) & dx.notna() & dy.notna(),
        np.sqrt(dx.pow(2) + dy.pow(2)) / dt,
        np.nan,
    )
    return df


def build_tracks(events: pd.DataFrame) -> pd.DataFrame:
    def safe_source_track_id(value: object) -> int | None:
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return None

    def agg_direction(series: pd.Series) -> str | None:
        non_null = series.dropna()
        return non_null.mode().iloc[0] if not non_null.empty else None

    def agg_path_length(group: pd.DataFrame) -> float | None:
        x = group["x_cord_m"].astype(float)
        y = group["y_cord_m"].astype(float)
        if len(group) < 2:
            return None
        return float(np.sqrt((x.diff().fillna(0).pow(2) + y.diff().fillna(0).pow(2))).sum())

    records = []
    for track_id, group in events.groupby("track_id", sort=False):
        group = group.sort_values("detection_time")
        first = group.iloc[0]
        last = group.iloc[-1]
        duration = (last["detection_time"] - first["detection_time"]).total_seconds()
        distance_proxy = agg_path_length(group)
        speed_proxy = distance_proxy / duration if duration and distance_proxy is not None else None
        vehicle_class = first["vehicle_class"] if "vehicle_class" in group.columns else map_vehicle_class(first["object_type"])
        records.append(
            {
                "track_id": str(track_id),
                "source_track_id": safe_source_track_id(track_id),
                "vehicle_class": vehicle_class,
                "first_seen_ts": first["detection_time"],
                "last_seen_ts": last["detection_time"],
                "duration_seconds": duration,
                "detections_count": int(len(group)),
                "start_centroid_x": first["x_cord_m"],
                "start_centroid_y": first["y_cord_m"],
                "end_centroid_x": last["x_cord_m"],
                "end_centroid_y": last["y_cord_m"],
                "direction": agg_direction(group["direction"]),
                "distance_proxy": distance_proxy,
                "speed_proxy": speed_proxy,
                "speed_kmh_estimated": None,
            }
        )
    return pd.DataFrame.from_records(records)
