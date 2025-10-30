"""Meta-heuristic outer loop for hyperparameter tuning and asset subset search.

This module provides a lightweight orchestration layer that leverages the
genetic algorithm utilities to explore different combinations of:

* Asset subsets (cardinality control)
* Risk-aversion and turnover penalties
* Optional turnover caps and cost scaling factors

Each candidate is evaluated by solving the convex mean-variance problem on the
selected subset, turning the GA into a meta-heuristic tuner on top of the core
optimizer. The best candidate (according to an arbitrary fitness metric) is
returned together with diagnostic information for downstream logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from itau_quant.optimization.core.mv_qp import MeanVarianceConfig, solve_mean_variance
from itau_quant.optimization.ga.genetic import GenerationSummary, GeneticRun, run_genetic_algorithm
from itau_quant.optimization.ga.population import Individual

__all__ = [
    "MetaheuristicResult",
    "metaheuristic_outer",
]


@dataclass(frozen=True)
class MetaheuristicResult:
    """Container for the outcome of the meta-heuristic search."""

    weights: pd.Series
    params: Mapping[str, Any]
    metrics: Mapping[str, Any]
    status: str
    fitness: float
    selected_assets: Sequence[str]
    history: Sequence[GenerationSummary] = field(default_factory=list)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


def _resolve_series(
    series: pd.Series | None,
    index: Sequence[str],
    *,
    fill_value: float,
) -> pd.Series:
    if series is None:
        return pd.Series(fill_value, index=index, dtype=float)
    return series.reindex(index).fillna(fill_value).astype(float)


def _build_candidate_config(
    base_config: MeanVarianceConfig,
    assets: Sequence[str],
    params: Mapping[str, Any],
) -> MeanVarianceConfig:
    universe_index = list(base_config.lower_bounds.index)
    assets_index = list(assets)

    lower_bounds = _resolve_series(base_config.lower_bounds, universe_index, fill_value=0.0).reindex(
        assets_index
    )
    upper_bounds = _resolve_series(base_config.upper_bounds, universe_index, fill_value=1.0).reindex(
        assets_index
    )
    previous_weights = _resolve_series(
        base_config.previous_weights, universe_index, fill_value=0.0
    ).reindex(assets_index)

    cost_vector = base_config.cost_vector
    if cost_vector is not None:
        cost_vector = cost_vector.reindex(assets_index).fillna(float(cost_vector.mean()))
        if "cost_scale" in params:
            cost_vector = cost_vector * float(params["cost_scale"])

    factor_loadings = None
    if base_config.factor_loadings is not None:
        factor_loadings = base_config.factor_loadings.reindex(index=assets_index)

    turnover_cap = params.get("tau", base_config.turnover_cap)
    if turnover_cap is not None:
        turnover_cap = float(turnover_cap)

    return MeanVarianceConfig(
        risk_aversion=float(params.get("lambda", base_config.risk_aversion)),
        turnover_penalty=float(params.get("eta", base_config.turnover_penalty)),
        turnover_cap=turnover_cap,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        previous_weights=previous_weights,
        cost_vector=cost_vector,
        budgets=base_config.budgets,
        solver=base_config.solver,
        solver_kwargs=base_config.solver_kwargs,
        risk_config=base_config.risk_config,
        factor_loadings=factor_loadings,
        ridge_penalty=base_config.ridge_penalty,
        target_vol=base_config.target_vol,
    )


def _build_full_weights(universe: Sequence[str], active_assets: Sequence[str], partial: pd.Series) -> pd.Series:
    weights = pd.Series(0.0, index=universe, dtype=float)
    weights.loc[list(active_assets)] = partial.reindex(active_assets).fillna(0.0).astype(float)
    total = float(weights.sum())
    if not np.isclose(total, 1.0):
        if total == 0.0:
            return weights
        weights /= total
    return weights


def _compute_penalties(
    metrics: Mapping[str, Any],
    *,
    turnover_target: tuple[float, float] | None,
    cardinality_target: tuple[int, int] | None,
    penalty_weights: Mapping[str, float],
) -> dict[str, float]:
    penalties: dict[str, float] = {}
    if turnover_target is not None:
        lo, hi = turnover_target
        turnover = float(metrics.get("turnover", 0.0))
        if turnover > hi:
            penalties["turnover"] = penalty_weights.get("turnover", 1.0) * (turnover - hi)
        elif turnover < lo:
            penalties["turnover"] = penalty_weights.get("turnover", 1.0) * (lo - turnover)

    if cardinality_target is not None:
        lo_k, hi_k = cardinality_target
        cardinality = int(metrics.get("cardinality", 0))
        if cardinality > hi_k:
            penalties["concentration"] = penalty_weights.get("cardinality", 1.0) * (cardinality - hi_k)
        elif cardinality < lo_k:
            penalties["concentration"] = penalty_weights.get("cardinality", 1.0) * (lo_k - cardinality)
    return penalties


def _evaluate_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    data = candidate["data"]
    base_config: MeanVarianceConfig = data["base_config"]
    mu: pd.Series = data["mu"]
    cov: pd.DataFrame = data["cov"]
    universe = list(data["universe"])

    assets: list[str] = list(candidate.get("assets") or [])
    if not assets:
        raise ValueError("candidate must select at least one asset")

    params = dict(candidate.get("params") or {})

    mu_subset = mu.reindex(assets).astype(float)
    cov_subset = cov.reindex(index=assets, columns=assets).astype(float)

    config = _build_candidate_config(base_config, assets, params)
    result = solve_mean_variance(mu_subset, cov_subset, config)

    weights_full = _build_full_weights(universe, assets, result.weights)
    previous_full = _resolve_series(base_config.previous_weights, universe, fill_value=0.0)
    full_turnover = float((weights_full - previous_full).abs().sum())

    variance = float(result.variance)
    if variance < 0:
        variance = float(np.clip(variance, a_min=0.0, a_max=None))
    sharpe = float(result.expected_return / np.sqrt(variance)) if variance > 0 else 0.0

    metrics = {
        "expected_return": float(result.expected_return),
        "variance": variance,
        "objective": float(result.objective_value),
        "turnover": full_turnover,
        "cost": float(result.cost),
        "sharpe": sharpe,
        "cardinality": int((weights_full.abs() > 1e-6).sum()),
    }

    penalties = _compute_penalties(
        metrics,
        turnover_target=data.get("turnover_target"),
        cardinality_target=data.get("cardinality_target"),
        penalty_weights=data.get("penalty_weights", {}),
    )

    diagnostics = {
        "solver_status": result.summary.status,
        "solver_runtime": result.summary.runtime,
        "params": params,
        "selected_assets": assets,
    }

    return {
        "weights": weights_full,
        "metrics": metrics,
        "status": result.summary.status,
        "penalties": penalties,
        "diagnostics": diagnostics,
    }


def _core_solver_factory(
    data: Mapping[str, Any],
) -> callable:
    def _core(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        enriched = dict(candidate)
        enriched["data"] = data
        return _evaluate_candidate(enriched)

    return _core


def _extract_selected_assets(individual: Individual, universe: Sequence[str]) -> list[str]:
    return individual.active_assets(universe)


def metaheuristic_outer(
    mu: pd.Series,
    cov: pd.DataFrame,
    base_config: MeanVarianceConfig,
    *,
    ga_config: Mapping[str, Any],
    turnover_target: tuple[float, float] | None = None,
    cardinality_target: tuple[int, int] | None = None,
    penalty_weights: Mapping[str, float] | None = None,
) -> MetaheuristicResult:
    """Run the GA-driven meta-heuristic search.

    Parameters
    ----------
    mu, cov : pd.Series, pd.DataFrame
        Estimated expected returns and covariance matrix.
    base_config : MeanVarianceConfig
        Baseline configuration used as template for each candidate solve.
    ga_config : Mapping[str, Any]
        Configuration dictionary forwarded to ``run_genetic_algorithm``.
    turnover_target : tuple[float, float], optional
        Desired turnover band; deviations are penalised in fitness.
    cardinality_target : tuple[int, int], optional
        Desired cardinality bounds for additional penalties.
    penalty_weights : Mapping[str, float], optional
        Weights applied to turnover/cardinality penalties.
    """

    mu = mu.astype(float)
    cov = cov.astype(float)
    universe = list(mu.index)

    data_payload = {
        "mu": mu,
        "cov": cov,
        "base_config": base_config,
        "universe": universe,
        "turnover_target": turnover_target,
        "cardinality_target": cardinality_target,
        "penalty_weights": dict(penalty_weights or {}),
    }

    core_solver = _core_solver_factory(data_payload)
    run: GeneticRun = run_genetic_algorithm(
        data={"universe": universe},
        config=ga_config,
        core_solver=core_solver,
    )

    best = run.best_result
    weights = best.weights
    if weights is None:
        weights = pd.Series(0.0, index=universe, dtype=float)
    else:
        weights = weights.reindex(universe).fillna(0.0).astype(float)

    metrics = dict(best.metrics)
    metrics.setdefault("fitness", float(best.fitness))

    params = dict(best.individual.params)
    selected_assets = _extract_selected_assets(best.individual, universe)

    diagnostics = dict(best.diagnostics)
    diagnostics.setdefault("individual_origin", best.individual.metadata.get("origin"))
    diagnostics.setdefault("history_length", len(run.history))
    diagnostics.setdefault("fitness", float(best.fitness))

    return MetaheuristicResult(
        weights=weights,
        params=params,
        metrics=metrics,
        status=best.status,
        fitness=float(best.fitness),
        selected_assets=selected_assets,
        history=list(run.history),
        diagnostics=diagnostics,
    )
