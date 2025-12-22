"""Helpers de caminho centralizando convenções do repositório.

Funções/Constantes
------------------
`find_project_root()`
    Caminha pelos diretórios ancestrais até encontrar ``pyproject.toml``,
    garantindo que runners executados fora da raiz ainda localizem os dados.

`PROJECT_ROOT`
    Resultado memoizado de ``find_project_root``.

`DATA_DIR`
    Diretório principal ``<root>/data``.

`RAW_DATA_DIR`, `PROCESSED_DATA_DIR`
    Subpastas padronizadas consumidas por loaders (`loader.py`, `storage.py`).
"""

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
