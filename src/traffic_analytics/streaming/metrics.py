from __future__ import annotations

from datetime import timedelta
import pandas as pd


VEHICLE_COLUMNS = ["car", "truck", "bus", "motorcycle", "bicycle", "person"]


def build_streaming_metrics(tracks: pd.DataFrame, camera_id: str, window_minutes: int) -> pd.DataFrame:
    if tracks.empty:
        return pd.DataFrame(columns=["camera_id", "window_start", "window_end", "window_granularity"])
    tracks = tracks.copy()
    tracks["window_start"] = tracks["first_seen_ts"].dt.floor(f"{window_minutes}min")
    grouped = tracks.groupby("window_start", dropna=False)
    rows = []
    max_count_reference = grouped.size().max()
    max_speed_reference = tracks["speed_proxy"].dropna().max()
    max_count_reference = int(max_count_reference) if pd.notna(max_count_reference) and max_count_reference else 1
    max_speed_reference = float(max_speed_reference) if pd.notna(max_speed_reference) and max_speed_reference else 1.0
    for window_start, group in grouped:
        counts = group["vehicle_class"].value_counts()
        total_count = int(len(group))
        avg_speed_proxy = group["speed_proxy"].dropna().mean()
        movement_score_avg = avg_speed_proxy
        occupancy_proxy = round(total_count / max_count_reference, 6)
        if avg_speed_proxy is None or pd.isna(avg_speed_proxy):
            congestion_proxy = occupancy_proxy
        else:
            congestion_proxy = round(occupancy_proxy * (1 - (avg_speed_proxy / max_speed_reference)), 6)
        heavy_vehicle_share = round(
            (counts.get("truck", 0) + counts.get("bus", 0)) / total_count,
            6,
        ) if total_count else None
        row = {
            "camera_id": camera_id,
            "window_start": window_start,
            "window_end": window_start + timedelta(minutes=window_minutes),
            "window_granularity": f"{window_minutes}m",
            "total_count": total_count,
            "car_count": int(counts.get("car", 0)),
            "truck_count": int(counts.get("truck", 0)),
            "bus_count": int(counts.get("bus", 0)),
            "motorcycle_count": int(counts.get("motorcycle", 0)),
            "bicycle_count": int(counts.get("bicycle", 0)),
            "person_count": int(counts.get("person", 0)),
            "avg_speed_proxy": None if pd.isna(avg_speed_proxy) else float(avg_speed_proxy),
            "avg_speed_kmh": None,
            "movement_score_avg": None if pd.isna(movement_score_avg) else float(movement_score_avg),
            "occupancy_proxy": occupancy_proxy,
            "congestion_proxy": congestion_proxy,
            "heavy_vehicle_share": heavy_vehicle_share,
        }
        rows.append(row)
    return pd.DataFrame(rows)

