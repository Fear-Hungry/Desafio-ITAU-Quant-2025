"""Arara Quant Lab (PRISM-R).

O código-fonte vive em `src/arara_quant/` e é consumido principalmente via:

- CLI: `poetry run arara-quant ...`
- Scripts: `poetry run python scripts/...`

Para guias de uso e arquitetura, veja `README.md` e `docs/README.md`.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover - depende de instalação do pacote
    __version__ = version("arara-quant-lab")
except PackageNotFoundError:  # pragma: no cover - fallback para ambiente sem install
    __version__ = "0.0.0"

__all__ = ["__version__"]
