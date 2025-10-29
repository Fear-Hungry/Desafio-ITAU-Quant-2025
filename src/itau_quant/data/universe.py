"""Helpers para o universo ARARA.

Recursos oferecidos
-------------------
`get_arara_metadata()`
    - Parser minimalista (sem PyYAML) para ler ``universe_metadata.yaml``.
    - Retorna ``Mapping[ticker, atributos]`` com limites de alocação, classes,
      moeda e outros metadados consumidos por otimizações.

`get_arara_universe()`
    - Preserva a ordenação do YAML ao produzir a lista base de tickers.
    - Utilizado como default por ``DataLoader`` e pela suíte de testes.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from typing import Any, Dict, List, Mapping

__all__ = ["get_arara_universe", "get_arara_metadata"]


def _coerce_value(raw: str) -> Any:
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    if raw.startswith("'") and raw.endswith("'"):
        return raw[1:-1]
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [_coerce_value(part.strip()) for part in inner.split(",")]
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _parse_simple_yaml(text: str) -> Dict[str, Dict[str, Any]]:
    data: Dict[str, Dict[str, Any]] = {}
    current_key: str | None = None
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" "):
            if not line.endswith(":"):
                raise ValueError(f"Linha {line_no}: esperado ':' no final da chave.")
            current_key = line[:-1].strip()
            if not current_key:
                raise ValueError(f"Linha {line_no}: nome de chave vazio.")
            if current_key in data:
                raise ValueError(f"Linha {line_no}: chave duplicada '{current_key}'.")
            data[current_key] = {}
        else:
            if current_key is None:
                raise ValueError(f"Linha {line_no}: indentação inesperada.")
            if not line.startswith("  "):
                raise ValueError(f"Linha {line_no}: indentação deve usar dois espaços.")
            stripped = line.strip()
            if ":" not in stripped:
                raise ValueError(f"Linha {line_no}: esperado par chave:valor.")
            sub_key, raw_value = stripped.split(":", 1)
            data[current_key][sub_key.strip()] = _coerce_value(raw_value.strip())
    return data


@lru_cache(maxsize=1)
def get_arara_metadata() -> Mapping[str, Mapping[str, Any]]:
    """Load ARARA universe metadata from the packaged YAML file."""
    with (
        resources.files(__package__)
        .joinpath("universe_metadata.yaml")
        .open("r", encoding="utf-8") as handle
    ):
        content = handle.read()
    return _parse_simple_yaml(content)


def get_arara_universe() -> List[str]:
    """Return the canonical ARARA tickers list preserving YAML order."""
    metadata = get_arara_metadata()
    return list(metadata.keys())
