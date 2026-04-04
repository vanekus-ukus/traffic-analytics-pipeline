from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from traffic_analytics.config.settings import Settings


@dataclass(slots=True)
class TrackStitchState:
    next_id: int = 1
    active_tracks: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=[
                "global_track_id",
                "vehicle_class",
                "last_seen_ts",
                "end_centroid_x",
                "end_centroid_y",
            ]
        )
    )


def _empty_active_tracks() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["global_track_id", "vehicle_class", "last_seen_ts", "end_centroid_x", "end_centroid_y"]
    )


def stitch_segment_tracks(
    segment_tracks: pd.DataFrame,
    state: TrackStitchState,
    settings: Settings,
) -> tuple[dict[str, str], int]:
    if segment_tracks.empty:
        return {}, 0

    if state.active_tracks.empty:
        state.active_tracks = _empty_active_tracks()

    mapping: dict[str, str] = {}
    stitched_count = 0
    now_ts = segment_tracks["last_seen_ts"].max()
    if pd.notna(now_ts):
        ttl_cutoff = now_ts - pd.Timedelta(seconds=settings.live_track_state_ttl_seconds)
        state.active_tracks = state.active_tracks.loc[
            state.active_tracks["last_seen_ts"].isna() | (state.active_tracks["last_seen_ts"] >= ttl_cutoff)
        ].copy()

    used_global_ids: set[str] = set()
    current_rows: list[dict[str, object]] = []
    active_tracks = state.active_tracks.copy()

    ordered_tracks = segment_tracks.sort_values(
        ["first_seen_ts", "detections_count"],
        ascending=[True, False],
    )

    for _, track in ordered_tracks.iterrows():
        local_track_id = str(track["track_id"])
        vehicle_class = track["vehicle_class"]
        best_global_id: str | None = None
        best_distance: float | None = None

        candidates = active_tracks.loc[active_tracks["vehicle_class"] == vehicle_class].copy()
        if not candidates.empty:
            candidates["time_gap"] = (
                pd.to_datetime(track["first_seen_ts"]) - pd.to_datetime(candidates["last_seen_ts"])
            ).dt.total_seconds()
            candidates = candidates.loc[
                (candidates["time_gap"] >= -1.0)
                & (candidates["time_gap"] <= settings.live_track_stitch_gap_seconds)
            ].copy()
            if not candidates.empty:
                candidates["distance_px"] = (
                    (candidates["end_centroid_x"].astype(float) - float(track["start_centroid_x"])).pow(2)
                    + (candidates["end_centroid_y"].astype(float) - float(track["start_centroid_y"])).pow(2)
                ) ** 0.5
                candidates = candidates.loc[
                    candidates["distance_px"] <= settings.live_track_stitch_distance_px
                ].sort_values(["distance_px", "time_gap"])

                for _, candidate in candidates.iterrows():
                    candidate_id = str(candidate["global_track_id"])
                    if candidate_id in used_global_ids:
                        continue
                    best_global_id = candidate_id
                    best_distance = float(candidate["distance_px"])
                    break

        if best_global_id is None:
            best_global_id = f"live_{state.next_id:08d}"
            state.next_id += 1
        else:
            stitched_count += 1

        used_global_ids.add(best_global_id)
        mapping[local_track_id] = best_global_id
        current_rows.append(
            {
                "global_track_id": best_global_id,
                "vehicle_class": vehicle_class,
                "last_seen_ts": track["last_seen_ts"],
                "end_centroid_x": track["end_centroid_x"],
                "end_centroid_y": track["end_centroid_y"],
                "distance_px": best_distance,
            }
        )

    current_df = pd.DataFrame(current_rows)
    if current_df.empty:
        return mapping, stitched_count

    remaining = active_tracks.loc[
        ~active_tracks["global_track_id"].astype(str).isin(current_df["global_track_id"].astype(str))
    ].copy()
    current_core = current_df[
        ["global_track_id", "vehicle_class", "last_seen_ts", "end_centroid_x", "end_centroid_y"]
    ].copy()
    if remaining.empty:
        state.active_tracks = current_core.reset_index(drop=True)
    else:
        state.active_tracks = pd.concat(
            [
                remaining[
                    ["global_track_id", "vehicle_class", "last_seen_ts", "end_centroid_x", "end_centroid_y"]
                ],
                current_core,
            ],
            ignore_index=True,
        )
    return mapping, stitched_count
