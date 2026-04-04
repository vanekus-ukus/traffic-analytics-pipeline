from __future__ import annotations

from pathlib import Path
import logging

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from traffic_analytics.config.settings import Settings

LOGGER = logging.getLogger(__name__)


def get_engine(settings: Settings) -> Engine:
    return create_engine(settings.sqlalchemy_url, future=True)


def run_sql_files(engine: Engine, sql_files: list[Path]) -> None:
    with engine.begin() as conn:
        for path in sql_files:
            LOGGER.info("Executing SQL file %s", path)
            conn.execute(text(path.read_text(encoding="utf-8")))

