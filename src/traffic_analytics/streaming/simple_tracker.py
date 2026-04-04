from __future__ import annotations

from dataclasses import dataclass
import itertools
import math


@dataclass(slots=True)
class Detection:
    vehicle_class: str
    confidence: float
    centroid_x: float
    centroid_y: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float


@dataclass(slots=True)
class TrackState:
    track_id: str
    vehicle_class: str
    centroid_x: float
    centroid_y: float
    missed_frames: int = 0


class SimpleCentroidTracker:
    def __init__(self, max_distance: float, max_missed_frames: int) -> None:
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self._tracks: dict[str, TrackState] = {}
        self._counter = itertools.count(1)

    @staticmethod
    def _distance(track: TrackState, detection: Detection) -> float:
        return math.sqrt(
            (track.centroid_x - detection.centroid_x) ** 2
            + (track.centroid_y - detection.centroid_y) ** 2
        )

    def update(self, detections: list[Detection]) -> list[tuple[str, Detection]]:
        assignments: list[tuple[str, Detection]] = []
        unmatched_tracks = set(self._tracks.keys())
        unmatched_detections = set(range(len(detections)))

        candidate_pairs: list[tuple[float, str, int]] = []
        for track_id, track in self._tracks.items():
            for detection_idx, detection in enumerate(detections):
                if track.vehicle_class != detection.vehicle_class:
                    continue
                distance = self._distance(track, detection)
                if distance <= self.max_distance:
                    candidate_pairs.append((distance, track_id, detection_idx))
        candidate_pairs.sort(key=lambda item: item[0])

        used_tracks: set[str] = set()
        used_detections: set[int] = set()
        for _, track_id, detection_idx in candidate_pairs:
            if track_id in used_tracks or detection_idx in used_detections:
                continue
            used_tracks.add(track_id)
            used_detections.add(detection_idx)
            unmatched_tracks.discard(track_id)
            unmatched_detections.discard(detection_idx)
            detection = detections[detection_idx]
            track = self._tracks[track_id]
            track.centroid_x = detection.centroid_x
            track.centroid_y = detection.centroid_y
            track.missed_frames = 0
            assignments.append((track_id, detection))

        for detection_idx in sorted(unmatched_detections):
            detection = detections[detection_idx]
            track_id = f"trk_{next(self._counter)}"
            self._tracks[track_id] = TrackState(
                track_id=track_id,
                vehicle_class=detection.vehicle_class,
                centroid_x=detection.centroid_x,
                centroid_y=detection.centroid_y,
            )
            assignments.append((track_id, detection))

        to_delete: list[str] = []
        for track_id in unmatched_tracks:
            track = self._tracks[track_id]
            track.missed_frames += 1
            if track.missed_frames > self.max_missed_frames:
                to_delete.append(track_id)
        for track_id in to_delete:
            del self._tracks[track_id]
        return assignments

