#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
PYTHONPATH="${ROOT_DIR}/src" .venv/bin/python -m traffic_analytics.streaming.export_debug_frames "$@"

