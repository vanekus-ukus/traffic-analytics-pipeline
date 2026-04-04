#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

python3 -m venv --clear --system-site-packages .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --no-deps ultralytics opencv-python-headless yt-dlp polars ultralytics-thop seaborn nbconvert

echo "Local environment prepared in .venv"

