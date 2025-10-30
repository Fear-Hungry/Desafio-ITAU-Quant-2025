"""Parâmetros padrão para as estratégias otimizadas.

O módulo centraliza *defaults* utilizados em testes e scripts, evitando a
duplicação dos mesmos números mágicos em vários pontos do código. Os valores
podem ser facilmente sobrescritos via ``merge_params``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "StrategyParams",
    "DEFAULT_PARAMS",
    "default_params",
    "merge_params",
]


@dataclass(frozen=True, slots=True)
class StrategyParams:
    """Container simples para os hiperparâmetros principais."""

    lambda_risk: float = 6.0
    eta_turnover: float = 0.5
    tau_turnover_cap: float = 0.20
    cardinality_min: int = 20
    cardinality_max: int = 35
    cvar_alpha: float = 0.95

    def to_dict(self) -> dict[str, Any]:
        return {
            "lambda_risk": self.lambda_risk,
            "eta_turnover": self.eta_turnover,
            "tau_turnover_cap": self.tau_turnover_cap,
            "cardinality_min": self.cardinality_min,
            "cardinality_max": self.cardinality_max,
            "cvar_alpha": self.cvar_alpha,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "StrategyParams":
        base = cls()
        base_dict = base.to_dict()
        merged = {**base_dict, **{k: mapping[k] for k in mapping if k in base_dict}}
        return cls(**merged)


DEFAULT_PARAMS = StrategyParams()


def default_params() -> StrategyParams:
    """Return a copy of the default parameters."""

    return StrategyParams(**DEFAULT_PARAMS.to_dict())


def merge_params(
    overrides: Mapping[str, Any] | None = None,
    *,
    base: StrategyParams | None = None,
) -> StrategyParams:
    """Merge ``overrides`` with ``base`` returning a new :class:`StrategyParams`.

    Parameters
    ----------
    overrides:
        Valores a substituir. Chaves desconhecidas geram ``KeyError`` para
        evitar erros silenciosos.
    base:
        Instância de referência; quando ``None`` usa :data:`DEFAULT_PARAMS`.
    """

    base_params = base or DEFAULT_PARAMS
    data = base_params.to_dict()

    if not overrides:
        return StrategyParams(**data)

    unknown = sorted(set(overrides) - set(data))
    if unknown:
        raise KeyError(f"Unknown parameter(s): {', '.join(unknown)}")

    data.update(overrides)
    return StrategyParams(**data)
