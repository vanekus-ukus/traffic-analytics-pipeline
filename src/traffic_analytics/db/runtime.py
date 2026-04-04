from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4
import json

from sqlalchemy import text
from sqlalchemy.engine import Engine


def start_pipeline_run(
    engine: Engine,
    pipeline_name: str,
    pipeline_mode: str,
    source_ref: str,
    config_snapshot: dict[str, Any],
) -> UUID:
    run_id = uuid4()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO ops.pipeline_runs (
                    run_id, pipeline_name, pipeline_mode, status,
                    started_at, source_ref, config_snapshot_json
                )
                VALUES (
                    :run_id, :pipeline_name, :pipeline_mode, 'running',
                    :started_at, :source_ref, CAST(:config_snapshot AS jsonb)
                )
                """
            ),
            {
                "run_id": str(run_id),
                "pipeline_name": pipeline_name,
                "pipeline_mode": pipeline_mode,
                "started_at": datetime.now(timezone.utc),
                "source_ref": source_ref,
                "config_snapshot": json.dumps(config_snapshot, ensure_ascii=False, default=str),
            },
        )
    return run_id


def finish_pipeline_run(
    engine: Engine,
    run_id: UUID,
    status: str,
    rows_written: int = 0,
    error_text: str | None = None,
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE ops.pipeline_runs
                SET finished_at = :finished_at,
                    status = :status,
                    rows_written = :rows_written,
                    error_text = :error_text
                WHERE run_id = :run_id
                """
            ),
            {
                "run_id": str(run_id),
                "finished_at": datetime.now(timezone.utc),
                "status": status,
                "rows_written": rows_written,
                "error_text": error_text,
            },
        )


def record_quality_check(
    engine: Engine,
    run_id: UUID,
    dataset_name: str,
    check_name: str,
    check_scope: str,
    status: str,
    actual_value: float | int | None = None,
    threshold_value: float | int | None = None,
    details_json: dict[str, Any] | None = None,
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO ops.data_quality_checks (
                    run_id, dataset_name, check_name, check_scope, status,
                    actual_value, threshold_value, details_json
                )
                VALUES (
                    :run_id, :dataset_name, :check_name, :check_scope, :status,
                    :actual_value, :threshold_value, CAST(:details_json AS jsonb)
                )
                """
            ),
            {
                "run_id": str(run_id),
                "dataset_name": dataset_name,
                "check_name": check_name,
                "check_scope": check_scope,
                "status": status,
                "actual_value": actual_value,
                "threshold_value": threshold_value,
                "details_json": json.dumps(details_json or {}, ensure_ascii=False, default=str),
            },
        )

