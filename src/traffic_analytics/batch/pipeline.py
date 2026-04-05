from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from traffic_analytics.common.logging_utils import configure_logging
from traffic_analytics.config.settings import get_settings
from traffic_analytics.db.engine import get_engine
from traffic_analytics.db.runtime import finish_pipeline_run, record_quality_check, start_pipeline_run

LOGGER = logging.getLogger(__name__)


def file_md5(path: Path) -> str:
    md5 = hashlib.md5()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def load_tracking_raw(engine, run_id, tracking_path: Path) -> int:
    df = pd.read_csv(tracking_path)
    for column in ["recording_at", "recording_end", "start_time", "end_time", "detection_time"]:
        df[column] = pd.to_datetime(df[column], errors="coerce", utc=True)
    df["source_file"] = str(tracking_path)
    df["run_id"] = str(run_id)
    df.to_sql("tracking_source_rows", engine, schema="raw", if_exists="append", index=False, method="multi")
    return len(df)


def load_traffic_raw(engine, run_id, traffic_path: Path) -> pd.DataFrame:
    df = pd.read_csv(traffic_path)
    df["timestamp_ts"] = pd.to_datetime(df["timestamp"], utc=True)
    df["date_value"] = pd.to_datetime(df["date"]).dt.date
    df["time_value"] = pd.to_datetime(df["time"], format="%H:%M:%S").dt.time
    df["source_file"] = str(traffic_path)
    df["run_id"] = str(run_id)
    df[
        [
            "run_id",
            "timestamp_ts",
            "date_value",
            "time_value",
            "avg_speed",
            "weather_type",
            "weather_code",
            "temperature",
            "precipitation",
            "intensity_30min",
            "cars",
            "trucks",
            "busses",
            "source_file",
        ]
    ].to_sql("traffic_30min_source", engine, schema="raw", if_exists="append", index=False, method="multi")
    return df


def register_batch_import(engine, run_id, dataset_name: str, source_path: Path, row_count: int) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO raw.batch_imports (
                    dataset_name, source_path, file_hash, row_count, status, run_id
                ) VALUES (
                    :dataset_name, :source_path, :file_hash, :row_count, 'loaded', :run_id
                )
                """
            ),
            {
                "dataset_name": dataset_name,
                "source_path": str(source_path),
                "file_hash": file_md5(source_path),
                "row_count": row_count,
                "run_id": str(run_id),
            },
        )


def build_batch_metrics(df: pd.DataFrame, camera_id: str) -> pd.DataFrame:
    batch = pd.DataFrame(
        {
            "camera_id": camera_id,
            "window_start": df["timestamp_ts"],
            "window_end": df["timestamp_ts"] + timedelta(minutes=30),
            "window_granularity": "30m",
            "total_count": df["intensity_30min"].astype(int),
            "car_count": df["cars"].astype(int),
            "truck_count": df["trucks"].astype(int),
            "bus_count": df["busses"].astype(int),
            "motorcycle_count": 0,
            "bicycle_count": 0,
            "person_count": 0,
            "avg_speed_proxy": df["avg_speed"].astype(float),
            "avg_speed_kmh": df["avg_speed"].astype(float),
            "movement_score_avg": df["avg_speed"].astype(float),
            "weather_type": df["weather_type"],
            "weather_code": df["weather_code"],
            "temperature": df["temperature"],
            "precipitation": df["precipitation"],
        }
    )
    max_intensity = max(batch["total_count"].max(), 1)
    max_speed = max(batch["avg_speed_kmh"].max(), 1.0)
    batch["occupancy_proxy"] = (batch["total_count"] / max_intensity).round(6)
    batch["congestion_proxy"] = (
        batch["occupancy_proxy"] * (1 - batch["avg_speed_kmh"] / max_speed)
    ).round(6)
    batch["heavy_vehicle_share"] = (
        (batch["truck_count"] + batch["bus_count"]) / batch["total_count"]
    ).round(6)
    return batch


def replace_batch_metrics(engine, run_id, batch_metrics: pd.DataFrame) -> int:
    payload = batch_metrics.copy()
    payload["run_id"] = str(run_id)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE dm.batch_metrics"))
    payload.to_sql("batch_metrics", engine, schema="dm", if_exists="append", index=False, method="multi")
    return len(payload)


def run() -> None:
    parser = argparse.ArgumentParser(description="Run batch pipeline.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)
    settings = get_settings()
    if settings.batch_traffic_csv is None:
        raise RuntimeError("TRAFFIC_BATCH_TRAFFIC is not configured.")
    if settings.tracking_fallback is None:
        raise RuntimeError("TRAFFIC_TRACKING_FALLBACK is not configured.")
    engine = get_engine(settings)
    config_snapshot = {
        "batch_traffic_csv": str(settings.batch_traffic_csv),
        "tracking_fallback": str(settings.tracking_fallback),
        "batch_camera_id": settings.batch_camera_id,
    }
    run_id = start_pipeline_run(
        engine,
        pipeline_name="traffic_batch_pipeline",
        pipeline_mode="batch",
        source_ref=str(settings.batch_traffic_csv),
        config_snapshot=config_snapshot,
    )
    rows_written = 0
    try:
        tracking_rows = load_tracking_raw(engine, run_id, settings.tracking_fallback)
        traffic_df = load_traffic_raw(engine, run_id, settings.batch_traffic_csv)
        register_batch_import(engine, run_id, "tracking_source_rows", settings.tracking_fallback, tracking_rows)
        register_batch_import(engine, run_id, "traffic_30min_source", settings.batch_traffic_csv, len(traffic_df))

        total_intensity_match = (
            traffic_df["cars"] + traffic_df["trucks"] + traffic_df["busses"] == traffic_df["intensity_30min"]
        ).all()
        record_quality_check(
            engine,
            run_id,
            dataset_name="traffic_30min_source",
            check_name="category_sum_matches_total",
            check_scope="dataset",
            status="passed" if total_intensity_match else "failed",
            actual_value=int(total_intensity_match),
            threshold_value=1,
        )
        suspicious_speed_rows = int((traffic_df["avg_speed"] > 150).sum())
        record_quality_check(
            engine,
            run_id,
            dataset_name="traffic_30min_source",
            check_name="avg_speed_outlier_count",
            check_scope="dataset",
            status="passed" if suspicious_speed_rows == 0 else "warning",
            actual_value=suspicious_speed_rows,
            threshold_value=0,
        )
        batch_metrics = build_batch_metrics(traffic_df, settings.batch_camera_id)
        rows_written = replace_batch_metrics(engine, run_id, batch_metrics)
        finish_pipeline_run(engine, run_id, status="success", rows_written=rows_written)
        LOGGER.info("Batch pipeline completed, rows_written=%s", rows_written)
    except Exception as exc:
        finish_pipeline_run(engine, run_id, status="failed", rows_written=rows_written, error_text=str(exc))
        raise


if __name__ == "__main__":
    run()
