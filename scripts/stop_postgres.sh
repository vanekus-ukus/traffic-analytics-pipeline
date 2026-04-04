#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi

TRAFFIC_PGDATA="${TRAFFIC_PGDATA:-.local_pg/data}"
PG_BIN_DIR="$(pg_config --bindir)"
PGDATA_PATH="${ROOT_DIR}/${TRAFFIC_PGDATA}"

if [[ -s "${PGDATA_PATH}/PG_VERSION" ]]; then
  "${PG_BIN_DIR}/pg_ctl" -D "${PGDATA_PATH}" stop -m fast >/dev/null || true
fi

