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
export TRAFFIC_DB_SOCKET_DIR="${TRAFFIC_DB_SOCKET_DIR:-/tmp/traffic_pg_socket}"
export TRAFFIC_PGDATA="${TRAFFIC_PGDATA:-.local_pg/data}"
export TRAFFIC_PGLOG="${TRAFFIC_PGLOG:-.local_pg/postgres.log}"

PG_BIN_DIR="$(pg_config --bindir)"
PGDATA_PATH="${ROOT_DIR}/${TRAFFIC_PGDATA}"
PGLOG_PATH="${ROOT_DIR}/${TRAFFIC_PGLOG}"
if [[ "${TRAFFIC_DB_SOCKET_DIR}" = /* ]]; then
  SOCKET_DIR="${TRAFFIC_DB_SOCKET_DIR}"
else
  SOCKET_DIR="${ROOT_DIR}/${TRAFFIC_DB_SOCKET_DIR}"
fi

mkdir -p "${PGDATA_PATH}" "$(dirname "${PGLOG_PATH}")" "${SOCKET_DIR}"

if [[ ! -s "${PGDATA_PATH}/PG_VERSION" ]]; then
  "${PG_BIN_DIR}/initdb" \
    --username="${TRAFFIC_DB_USER}" \
    --auth=trust \
    --encoding=UTF8 \
    --locale=C.UTF-8 \
    -D "${PGDATA_PATH}" >/dev/null
fi

if ! "${PG_BIN_DIR}/pg_ctl" -D "${PGDATA_PATH}" status >/dev/null 2>&1; then
  "${PG_BIN_DIR}/pg_ctl" \
    -D "${PGDATA_PATH}" \
    -l "${PGLOG_PATH}" \
    -o "-c listen_addresses='' -c port=${TRAFFIC_DB_PORT} -c unix_socket_directories='${SOCKET_DIR}'" \
    start >/dev/null
fi

if ! psql -h "${SOCKET_DIR}" -p "${TRAFFIC_DB_PORT}" -U "${TRAFFIC_DB_USER}" -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='${TRAFFIC_DB_NAME}'" | grep -q 1; then
  createdb -h "${SOCKET_DIR}" -p "${TRAFFIC_DB_PORT}" -U "${TRAFFIC_DB_USER}" "${TRAFFIC_DB_NAME}"
fi

echo "PostgreSQL is running on socket ${SOCKET_DIR}, database=${TRAFFIC_DB_NAME}"
