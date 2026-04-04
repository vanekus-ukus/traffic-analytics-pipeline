from __future__ import annotations

from pathlib import Path
import argparse

from traffic_analytics.common.logging_utils import configure_logging
from traffic_analytics.config.settings import get_settings
from traffic_analytics.db.engine import get_engine, run_sql_files


def build_sql_paths(root: Path) -> list[Path]:
    sql_dir = root / "sql" / "init"
    return [
        sql_dir / "001_schemas.sql",
        sql_dir / "002_raw_tables.sql",
        sql_dir / "003_core_tables.sql",
        sql_dir / "004_dm_tables.sql",
        sql_dir / "005_views.sql",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize database schema.")
    parser.parse_args()
    configure_logging()
    settings = get_settings()
    root = Path(__file__).resolve().parents[3]
    engine = get_engine(settings)
    run_sql_files(engine, build_sql_paths(root))


if __name__ == "__main__":
    main()
