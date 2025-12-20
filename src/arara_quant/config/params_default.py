"""Parâmetros padrão para as estratégias e experimentos.

O módulo centraliza *defaults* utilizados em testes, scripts e loaders de YAML,
evitando a duplicação dos mesmos números mágicos em vários pontos do código. Os
valores podem ser facilmente sobrescritos via ``merge_params`` ou via arquivos
de configuração YAML, mantendo uma única fonte para defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .constants import TRADING_DAYS_IN_YEAR

__all__ = [
    "StrategyParams",
    "DEFAULT_PARAMS",
    "default_params",
    "merge_params",
    "MuEstimatorDefaults",
    "SigmaEstimatorDefaults",
    "GraphicalLassoDefaults",
    "OptimizerYamlDefaults",
    "WalkforwardDefaults",
    "DEFAULT_OPTIMIZER_YAML",
    "DEFAULT_WALKFORWARD",
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
    def from_mapping(cls, mapping: Mapping[str, Any]) -> StrategyParams:
        base = cls()
        base_dict = base.to_dict()
        merged = {**base_dict, **{k: mapping[k] for k in mapping if k in base_dict}}
        return cls(**merged)


DEFAULT_PARAMS = StrategyParams()


@dataclass(frozen=True, slots=True)
class MuEstimatorDefaults:
    """Defaults for μ estimators used in YAML-driven workflows."""

    method: str = "simple"
    window_days: int = 0
    huber_delta: float = 1.5
    shrink_strength: float = 0.5
    student_t_nu: float = 5.0
    prior: float = 0.0


@dataclass(frozen=True, slots=True)
class GraphicalLassoDefaults:
    """Defaults for Graphical Lasso covariance estimation."""

    alpha: float = 0.01
    max_iter: int = 200
    tol: float = 1e-4
    enet_tol: float = 1e-4
    mode: str = "cd"
    sparsity_tol: float = 1e-6


@dataclass(frozen=True, slots=True)
class SigmaEstimatorDefaults:
    """Defaults for Σ estimators used in YAML-driven workflows."""

    method: str = "ledoit_wolf"
    window_days: int = 0
    nonlinear: bool = False
    assume_centered: bool = False
    graphical_lasso: GraphicalLassoDefaults = field(
        default_factory=GraphicalLassoDefaults
    )


@dataclass(frozen=True, slots=True)
class OptimizerYamlDefaults:
    """Defaults applied when optimizer YAML omits fields."""

    objective: str = "mean_variance"
    long_only: bool = True
    risk_aversion: float = 5.0
    turnover_penalty: float = 0.0
    min_weight: float = 0.0
    max_weight: float = 1.0
    returns_filename: str = "returns_arara.parquet"
    linear_costs_bps: float = 0.0
    estimators_mu: MuEstimatorDefaults = field(default_factory=MuEstimatorDefaults)
    estimators_sigma: SigmaEstimatorDefaults = field(
        default_factory=SigmaEstimatorDefaults
    )


@dataclass(frozen=True, slots=True)
class WalkforwardDefaults:
    """Defaults for walk-forward backtests."""

    train_days: int = TRADING_DAYS_IN_YEAR
    test_days: int = 21
    purge_days: int = 0
    embargo_days: int = 0
    evaluation_horizons: tuple[int, ...] = (21, 63, 126)


DEFAULT_OPTIMIZER_YAML = OptimizerYamlDefaults()
DEFAULT_WALKFORWARD = WalkforwardDefaults()


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
