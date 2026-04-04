from __future__ import annotations

RAW_TO_CANONICAL = {
    "light car": "car",
    "medium car": "car",
    "heavy car": "truck",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "buss": "bus",
    "bicycle": "bicycle",
    "bike": "bicycle",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "pedestrian": "person",
    "person": "person",
}


def map_vehicle_class(raw_value: str | None) -> str:
    if not raw_value:
        return "unknown"
    normalized = raw_value.strip().lower()
    return RAW_TO_CANONICAL.get(normalized, normalized.replace(" ", "_"))

