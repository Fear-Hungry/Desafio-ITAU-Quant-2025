"""Public entry points for optimisation routines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from itau_quant.config import Settings, get_settings
from itau_quant.estimators import cov as covariance_estimators
from itau_quant.estimators import mu as mean_estimators
from itau_quant.optimization.core.mv_qp import (
    MeanVarianceConfig,
    MeanVarianceResult,
    solve_mean_variance,
)
from itau_quant.optimization.core.solver_utils import SolverSummary
from itau_quant.utils.data_loading import read_dataframe, read_vector
from itau_quant.utils.yaml_loader import load_yaml_text

DEFAULT_OPTIMIZER_CONFIG = "optimizer_example.yaml"

__all__ = [
    "OptimizationResult",
    "resolve_config_path",
    "run_optimizer",
]


@dataclass(frozen=True)
class OptimizerConfig:
    config_path: Path
    base_currency: str
    risk_aversion: float
    turnover_penalty: float
    turnover_cap: float | None
    min_weight: float
    max_weight: float
    linear_costs_bps: float | Mapping[str, float]
    mu_config: Mapping[str, Any]
    sigma_config: Mapping[str, Any]
    returns_path: Path
    previous_weights: pd.Series | None
    solver: str | None
    solver_kwargs: Mapping[str, Any]
    risk_config: Mapping[str, Any] | None = None


@dataclass(slots=True)
class OptimizationResult:
    config_path: Path
    environment: str
    base_currency: str
    dry_run: bool
    weights: pd.Series | None = None
    metrics: dict[str, float] | None = None
    turnover: float | None = None
    cost: float | None = None
    summary: SolverSummary | None = None
    notes: list[str] = field(default_factory=list)

    def status(self) -> str:
        if self.dry_run:
            return "preview"
        if self.summary and self.summary.is_optimal():
            return "optimal"
        return self.summary.status if self.summary else "unknown"

    def to_dict(self, include_weights: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "config_path": str(self.config_path),
            "environment": self.environment,
            "base_currency": self.base_currency,
            "dry_run": self.dry_run,
            "status": self.status(),
        }
        if self.metrics is not None:
            payload["metrics"] = dict(self.metrics)
        if self.turnover is not None:
            payload["turnover"] = float(self.turnover)
        if self.cost is not None:
            payload["cost"] = float(self.cost)
        if self.summary is not None:
            payload["solver"] = {
                "name": self.summary.solver,
                "status": self.summary.status,
                "objective": self.summary.value,
                "runtime": self.summary.runtime,
            }
        if self.notes:
            payload["notes"] = list(self.notes)
        if include_weights and self.weights is not None:
            payload["weights"] = [
                {"asset": asset, "weight": float(weight)}
                for asset, weight in self.weights.items()
            ]
        return payload


def resolve_config_path(
    config_path: str | Path | None,
    *,
    settings: Settings | None = None,
) -> Path:
    settings = settings or get_settings()

    if config_path is None:
        candidate = settings.configs_dir / DEFAULT_OPTIMIZER_CONFIG
    else:
        candidate = Path(config_path)
        if not candidate.is_absolute():
            in_configs = settings.configs_dir / candidate
            candidate = in_configs if in_configs.exists() else settings.project_root / candidate

    candidate = candidate.expanduser().resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"optimizer config not found: {candidate}")
    return candidate


def run_optimizer(
    config_path: str | Path | None = None,
    *,
    dry_run: bool = True,
    settings: Settings | None = None,
) -> OptimizationResult:
    settings = settings or get_settings()
    resolved = resolve_config_path(config_path, settings=settings)
    optimizer_config = _load_config(resolved, settings)

    result = OptimizationResult(
        config_path=resolved,
        environment=settings.environment,
        base_currency=optimizer_config.base_currency,
        dry_run=dry_run,
    )

    if dry_run:
        return result

    returns = _load_returns(optimizer_config)
    mu_series, cov_matrix = _estimate_inputs(returns, optimizer_config)

    lower = pd.Series(optimizer_config.min_weight, index=mu_series.index, dtype=float)
    upper = pd.Series(optimizer_config.max_weight, index=mu_series.index, dtype=float)

    previous_weights = optimizer_config.previous_weights
    if previous_weights is None:
        previous_weights = pd.Series(0.0, index=mu_series.index, dtype=float)
    else:
        previous_weights = previous_weights.reindex(mu_series.index, fill_value=0.0).astype(float)

    cost_vector = _build_cost_vector(optimizer_config.linear_costs_bps, mu_series.index)

    mv_config = MeanVarianceConfig(
        risk_aversion=optimizer_config.risk_aversion,
        turnover_penalty=optimizer_config.turnover_penalty,
        turnover_cap=optimizer_config.turnover_cap,
        lower_bounds=lower,
        upper_bounds=upper,
        previous_weights=previous_weights,
        cost_vector=cost_vector,
        solver=optimizer_config.solver,
        solver_kwargs=optimizer_config.solver_kwargs,
        risk_config=optimizer_config.risk_config,
    )

    mv_result = solve_mean_variance(mu_series, cov_matrix, mv_config)

    metrics = {
        "expected_return": mv_result.expected_return,
        "variance": mv_result.variance,
        "objective_value": mv_result.objective_value,
    }

    result.weights = mv_result.weights
    result.metrics = metrics
    result.turnover = mv_result.turnover
    result.cost = mv_result.cost
    result.summary = mv_result.summary
    result.dry_run = False

    if mv_result.summary.status == "infeasible":
        result.notes.append("Solver reported infeasible problem")

    return result


def _load_config(path: Path, settings: Settings) -> OptimizerConfig:
    raw = load_yaml_text(path.read_text(encoding="utf-8"))

    base_currency = raw.get("base_currency", settings.base_currency)
    optimizer_section = raw.get("optimizer", {})

    risk_aversion = float(optimizer_section.get("lambda", 5.0))
    turnover_penalty = float(optimizer_section.get("eta", 0.0))
    turnover_cap = optimizer_section.get("tau")
    if turnover_cap is not None:
        turnover_cap = float(turnover_cap)

    min_weight = float(optimizer_section.get("min_weight", 0.0))
    max_weight = float(optimizer_section.get("max_weight", 1.0))

    linear_costs = raw.get("estimators", {}).get("costs", {}).get("linear_bps", 0.0)

    mu_config = raw.get("estimators", {}).get("mu", {})
    sigma_config = raw.get("estimators", {}).get("sigma", {})
    risk_section = optimizer_section.get("risk") or optimizer_section.get("risk_constraints")
    risk_config = None
    if isinstance(risk_section, Mapping):
        risk_config = {k: v for k, v in risk_section.items()}

    data_section = raw.get("data", {})
    returns_path_raw = data_section.get("returns") or data_section.get("returns_path")
    if returns_path_raw is None:
        returns_path = settings.data_dir / "processed" / "returns_arara.parquet"
    else:
        returns_path = _resolve_relative_path(returns_path_raw, base=path.parent, settings=settings, must_exist=False)

    previous_weights: pd.Series | None = None
    state_section = raw.get("state", {})
    prev_data = state_section.get("previous_weights")
    if isinstance(prev_data, str):
        prev_path = _resolve_relative_path(prev_data, base=path.parent, settings=settings, must_exist=False)
        if prev_path.exists():
            previous_weights = read_vector(prev_path)
    elif isinstance(prev_data, Mapping):
        previous_weights = pd.Series({k: float(v) for k, v in prev_data.items()}, dtype=float)

    solver = optimizer_section.get("solver")
    solver_kwargs = optimizer_section.get("solver_kwargs", {})

    return OptimizerConfig(
        config_path=path,
        base_currency=base_currency,
        risk_aversion=risk_aversion,
        turnover_penalty=turnover_penalty,
        turnover_cap=turnover_cap,
        min_weight=min_weight,
        max_weight=max_weight,
        linear_costs_bps=linear_costs,
        mu_config=mu_config,
        sigma_config=sigma_config,
        returns_path=returns_path,
        previous_weights=previous_weights,
        solver=solver,
        solver_kwargs=solver_kwargs,
        risk_config=risk_config,
    )


def _resolve_relative_path(value: str | Path, *, base: Path, settings: Settings, must_exist: bool = True) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (base / candidate).resolve()
    if not candidate.exists():
        fallback = (settings.project_root / Path(value)).resolve()
        if fallback.exists():
            candidate = fallback
    if must_exist and not candidate.exists():
        raise FileNotFoundError(f"Data file not found: {candidate}")
    return candidate


def _load_returns(config: OptimizerConfig) -> pd.DataFrame:
    data = read_dataframe(config.returns_path)
    if isinstance(data, pd.Series):
        data = data.to_frame()
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    return data.astype(float)


def _estimate_inputs(returns: pd.DataFrame, config: OptimizerConfig) -> tuple[pd.Series, pd.DataFrame]:
    mu_config = config.mu_config
    sigma_config = config.sigma_config

    mu_window = int(mu_config.get("window_days", 0) or 0)
    sigma_window = int(sigma_config.get("window_days", 0) or 0)

    data_for_mu = returns.tail(mu_window) if mu_window > 0 else returns
    data_for_sigma = returns.tail(sigma_window) if sigma_window > 0 else returns

    mu_method = (mu_config.get("method") or "simple").lower()
    if mu_method in {"simple", "mean"}:
        mu_series = mean_estimators.mean_return(data_for_mu, method="simple")
    elif mu_method == "geometric":
        mu_series = mean_estimators.mean_return(data_for_mu, method="geometric")
    elif mu_method == "huber":
        delta = float(mu_config.get("delta", 1.5))
        mu_series, _ = mean_estimators.huber_mean(data_for_mu, c=delta)
    elif mu_method == "student_t":
        nu = float(mu_config.get("nu", 5.0))
        mu_series = mean_estimators.student_t_mean(data_for_mu, nu=nu)
    else:
        raise ValueError(f"Unsupported mean estimator '{mu_method}'")

    sigma_method = (sigma_config.get("method") or "ledoit_wolf").lower()
    if sigma_method == "ledoit_wolf":
        nonlinear = bool(sigma_config.get("nonlinear", False))
        if nonlinear:
            cov_matrix = covariance_estimators.nonlinear_shrinkage(data_for_sigma)
        else:
            cov_matrix, _ = covariance_estimators.ledoit_wolf_shrinkage(data_for_sigma)
    elif sigma_method == "sample":
        cov_matrix = covariance_estimators.sample_cov(data_for_sigma)
    elif sigma_method == "tyler":
        cov_matrix = covariance_estimators.tyler_m_estimator(data_for_sigma)
    else:
        raise ValueError(f"Unsupported covariance estimator '{sigma_method}'")

    mu_series = mu_series.astype(float)
    cov_matrix = cov_matrix.astype(float)
    cov_matrix = cov_matrix.reindex(index=mu_series.index, columns=mu_series.index).fillna(0.0)

    return mu_series, cov_matrix


def _build_cost_vector(costs: float | Mapping[str, float], index: pd.Index) -> pd.Series | None:
    if isinstance(costs, Mapping):
        series = pd.Series({str(k): float(v) for k, v in costs.items()}, dtype=float)
        return series.reindex(index).fillna(series.mean() if not series.empty else 0.0) / 10_000.0
    scalar = float(costs)
    if scalar == 0:
        return None
    return pd.Series(scalar / 10_000.0, index=index, dtype=float)
