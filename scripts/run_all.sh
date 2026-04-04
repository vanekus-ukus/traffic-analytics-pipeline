#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

bash scripts/start_postgres.sh
bash scripts/init_db.sh
bash scripts/run_batch.sh
bash scripts/run_streaming.sh

echo "Pipelines finished successfully."
