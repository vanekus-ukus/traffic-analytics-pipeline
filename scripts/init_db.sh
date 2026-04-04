#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi

export TRAFFIC_DB_HOST="${TRAFFIC_DB_HOST:-/tmp/traffic_pg_socket}"
export TRAFFIC_DB_PORT="${TRAFFIC_DB_PORT:-55432}"
export TRAFFIC_DB_NAME="${TRAFFIC_DB_NAME:-transport_analytics}"
export TRAFFIC_DB_USER="${TRAFFIC_DB_USER:-transport_user}"

for sql_file in \
  "${ROOT_DIR}/sql/init/001_schemas.sql" \
  "${ROOT_DIR}/sql/init/002_raw_tables.sql" \
  "${ROOT_DIR}/sql/init/003_core_tables.sql" \
  "${ROOT_DIR}/sql/init/004_dm_tables.sql" \
  "${ROOT_DIR}/sql/init/005_views.sql"
do
  psql -h "${TRAFFIC_DB_HOST}" -p "${TRAFFIC_DB_PORT}" -U "${TRAFFIC_DB_USER}" -d "${TRAFFIC_DB_NAME}" -f "${sql_file}"
done

echo "Database schema initialized."
