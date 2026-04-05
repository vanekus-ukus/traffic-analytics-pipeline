from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from traffic_analytics.common.env import load_dotenv


def _parse_source_pool(raw_value: str) -> tuple[str, ...]:
    items = [item.strip() for item in raw_value.split(",") if item.strip()]
    return tuple(items)


@dataclass(slots=True)
class Settings:
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str
    db_socket_dir: Path
    pgdata: Path
    pglog: Path
    stream_source: str
    stream_source_pool: tuple[str, ...]
    vk_url: str
    video_fallback: Path
    tracking_fallback: Path
    batch_traffic_csv: Path
    artifacts_dir: Path
    stream_window_minutes: int
    batch_camera_id: str
    stream_camera_id: str
    enable_network: bool
    yolo_model_path: Path
    yolo_confidence: float
    yolo_imgsz: int
    stream_target_fps: int
    stream_max_seconds: int
    stream_max_missed_frames: int
    vk_cache_path: Path
    live_segment_dir: Path
    live_segment_seconds: int
    live_runtime_seconds: int
    live_capture_timeout_seconds: int
    live_keep_segments: bool
    live_resolve_each_cycle: bool
    live_retry_backoff_seconds: int
    live_max_consecutive_failures: int
    live_track_stitch_enabled: bool
    live_track_stitch_distance_px: float
    live_track_stitch_gap_seconds: float
    live_track_state_ttl_seconds: float
    motion_roi_enabled: bool
    motion_mask_threshold: float
    motion_bbox_min_ratio: float
    motion_roi_x_min_ratio: float
    motion_roi_x_max_ratio: float
    motion_roi_y_min_ratio: float
    motion_roi_y_max_ratio: float
    edge_border_ratio: float
    edge_motion_relax_factor: float
    edge_track_min_detections: int
    edge_track_min_distance_px: float
    edge_track_min_confidence: float
    track_min_detections: int
    track_min_distance_px: float
    track_min_confidence: float
    max_bbox_area_ratio: float
    vehicle_min_aspect_ratio: float
    vehicle_max_aspect_ratio: float
    upper_zone_vehicle_bottom_ratio: float
    upper_zone_motion_ratio: float
    static_track_min_duration_sec: float
    static_track_max_distance_px: float

    @property
    def sqlalchemy_url(self) -> str:
        if self.db_host.startswith("/") or self.db_host.startswith("."):
            return (
                f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
                f"@/{self.db_name}?host={self.db_host}&port={self.db_port}"
            )
        return (
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        db_host=os.getenv("TRAFFIC_DB_HOST", "/tmp/traffic_pg_socket"),
        db_port=int(os.getenv("TRAFFIC_DB_PORT", "55432")),
        db_name=os.getenv("TRAFFIC_DB_NAME", "transport_analytics"),
        db_user=os.getenv("TRAFFIC_DB_USER", "transport_user"),
        db_password=os.getenv("TRAFFIC_DB_PASSWORD", "transport_pass"),
        db_socket_dir=Path(os.getenv("TRAFFIC_DB_SOCKET_DIR", "/tmp/traffic_pg_socket")),
        pgdata=Path(os.getenv("TRAFFIC_PGDATA", ".local_pg/data")),
        pglog=Path(os.getenv("TRAFFIC_PGLOG", ".local_pg/postgres.log")),
        stream_source=os.getenv(
            "TRAFFIC_STREAM_SOURCE",
            os.getenv("TRAFFIC_VK_URL", ""),
        ),
        stream_source_pool=_parse_source_pool(os.getenv("TRAFFIC_STREAM_SOURCE_POOL", "")),
        vk_url=os.getenv("TRAFFIC_VK_URL", ""),
        video_fallback=Path(os.getenv("TRAFFIC_VIDEO_FALLBACK", "data/input.mp4")),
        tracking_fallback=Path(
            os.getenv("TRAFFIC_TRACKING_FALLBACK", "data/tracking.csv")
        ),
        batch_traffic_csv=Path(
            os.getenv("TRAFFIC_BATCH_TRAFFIC", "data/traffic.csv")
        ),
        artifacts_dir=Path(os.getenv("TRAFFIC_ARTIFACTS_DIR", "artifacts")),
        stream_window_minutes=int(os.getenv("TRAFFIC_STREAM_WINDOW_MINUTES", "1")),
        batch_camera_id=os.getenv("TRAFFIC_BATCH_CAMERA_ID", "historical_camera_1"),
        stream_camera_id=os.getenv("TRAFFIC_STREAM_CAMERA_ID", "stream_camera_1"),
        enable_network=os.getenv("TRAFFIC_ENABLE_NETWORK", "0") == "1",
        yolo_model_path=Path(os.getenv("TRAFFIC_YOLO_MODEL_PATH", "yolov8n.pt")),
        yolo_confidence=float(os.getenv("TRAFFIC_YOLO_CONFIDENCE", "0.25")),
        yolo_imgsz=int(os.getenv("TRAFFIC_YOLO_IMGSZ", "960")),
        stream_target_fps=int(os.getenv("TRAFFIC_STREAM_TARGET_FPS", "10")),
        stream_max_seconds=int(os.getenv("TRAFFIC_STREAM_MAX_SECONDS", "180")),
        stream_max_missed_frames=int(os.getenv("TRAFFIC_STREAM_MAX_MISSED_FRAMES", "10")),
        vk_cache_path=Path(os.getenv("TRAFFIC_VK_CACHE_PATH", "data_cache/vk_preview.mp4")),
        live_segment_dir=Path(os.getenv("LIVE_SEGMENT_DIR", "artifacts/live_buffer")),
        live_segment_seconds=int(os.getenv("LIVE_SEGMENT_SECONDS", "10")),
        live_runtime_seconds=int(os.getenv("LIVE_RUNTIME_SECONDS", "300")),
        live_capture_timeout_seconds=int(os.getenv("LIVE_CAPTURE_TIMEOUT_SECONDS", "25")),
        live_keep_segments=os.getenv("LIVE_KEEP_SEGMENTS", "0") == "1",
        live_resolve_each_cycle=os.getenv("LIVE_RESOLVE_EACH_CYCLE", "1") == "1",
        live_retry_backoff_seconds=int(os.getenv("LIVE_RETRY_BACKOFF_SECONDS", "2")),
        live_max_consecutive_failures=int(os.getenv("LIVE_MAX_CONSECUTIVE_FAILURES", "6")),
        live_track_stitch_enabled=os.getenv("LIVE_TRACK_STITCH_ENABLED", "1") == "1",
        live_track_stitch_distance_px=float(os.getenv("LIVE_TRACK_STITCH_DISTANCE_PX", "120")),
        live_track_stitch_gap_seconds=float(os.getenv("LIVE_TRACK_STITCH_GAP_SECONDS", "20")),
        live_track_state_ttl_seconds=float(os.getenv("LIVE_TRACK_STATE_TTL_SECONDS", "60")),
        motion_roi_enabled=os.getenv("TRAFFIC_MOTION_ROI_ENABLED", "1") == "1",
        motion_mask_threshold=float(os.getenv("TRAFFIC_MOTION_MASK_THRESHOLD", "0.03")),
        motion_bbox_min_ratio=float(os.getenv("TRAFFIC_MOTION_BBOX_MIN_RATIO", "0.02")),
        motion_roi_x_min_ratio=float(os.getenv("TRAFFIC_MOTION_ROI_X_MIN_RATIO", "0.0")),
        motion_roi_x_max_ratio=float(os.getenv("TRAFFIC_MOTION_ROI_X_MAX_RATIO", "1.0")),
        motion_roi_y_min_ratio=float(os.getenv("TRAFFIC_MOTION_ROI_Y_MIN_RATIO", "0.0")),
        motion_roi_y_max_ratio=float(os.getenv("TRAFFIC_MOTION_ROI_Y_MAX_RATIO", "1.0")),
        edge_border_ratio=float(os.getenv("TRAFFIC_EDGE_BORDER_RATIO", "0.04")),
        edge_motion_relax_factor=float(os.getenv("TRAFFIC_EDGE_MOTION_RELAX_FACTOR", "0.25")),
        edge_track_min_detections=int(os.getenv("TRAFFIC_EDGE_TRACK_MIN_DETECTIONS", "1")),
        edge_track_min_distance_px=float(os.getenv("TRAFFIC_EDGE_TRACK_MIN_DISTANCE_PX", "4")),
        edge_track_min_confidence=float(os.getenv("TRAFFIC_EDGE_TRACK_MIN_CONFIDENCE", "0.18")),
        track_min_detections=int(os.getenv("TRAFFIC_TRACK_MIN_DETECTIONS", "1")),
        track_min_distance_px=float(os.getenv("TRAFFIC_TRACK_MIN_DISTANCE_PX", "10")),
        track_min_confidence=float(os.getenv("TRAFFIC_TRACK_MIN_CONFIDENCE", "0.25")),
        max_bbox_area_ratio=float(os.getenv("TRAFFIC_MAX_BBOX_AREA_RATIO", "0.35")),
        vehicle_min_aspect_ratio=float(os.getenv("VEHICLE_MIN_ASPECT_RATIO", "0.35")),
        vehicle_max_aspect_ratio=float(os.getenv("VEHICLE_MAX_ASPECT_RATIO", "4.5")),
        upper_zone_vehicle_bottom_ratio=float(os.getenv("UPPER_ZONE_VEHICLE_BOTTOM_RATIO", "0.25")),
        upper_zone_motion_ratio=float(os.getenv("UPPER_ZONE_MOTION_RATIO", "0.12")),
        static_track_min_duration_sec=float(os.getenv("TRAFFIC_STATIC_TRACK_MIN_DURATION_SEC", "5")),
        static_track_max_distance_px=float(os.getenv("TRAFFIC_STATIC_TRACK_MAX_DISTANCE_PX", "20")),
    )
