"""Canonical paths for data artefacts."""

from __future__ import annotations

from pathlib import Path


def find_project_root() -> Path:
    """Locate the project root by searching for ``pyproject.toml``."""
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd()


PROJECT_ROOT: Path = find_project_root()
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
