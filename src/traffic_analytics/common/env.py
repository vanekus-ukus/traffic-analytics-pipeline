from __future__ import annotations

from pathlib import Path
import os


def load_dotenv(env_path: str | Path = ".env") -> None:
    path = Path(env_path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        cleaned = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), cleaned)
