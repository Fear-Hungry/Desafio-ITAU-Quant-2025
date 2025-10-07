from __future__ import annotations

from pathlib import Path


def find_project_root() -> Path:
    """Resolve a raiz do projeto procurando por pyproject.toml.

    Retorna cwd como fallback para ambientes não versionados.
    """
    p = Path(__file__).resolve().parent
    for parent in [p, *p.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


PROJECT_ROOT: Path = find_project_root()
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

