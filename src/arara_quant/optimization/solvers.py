"""Public entry points for optimisation routines."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from arara_quant.config import Settings, get_settings
from arara_quant.config.params_default import DEFAULT_OPTIMIZER_YAML
from arara_quant.estimators import cov as covariance_estimators
from arara_quant.estimators import mu as mean_estimators
from arara_quant.optimization.core.cvar_lp import CvarConfig, solve_cvar_lp
from arara_quant.optimization.core.mv_qp import (
    MeanVarianceConfig,
    solve_mean_variance,
)
from arara_quant.optimization.core.solver_utils import SolverSummary
from arara_quant.optimization.heuristics.metaheuristic import (
    MetaheuristicResult,
    metaheuristic_outer,
)
from arara_quant.risk.budgets import RiskBudget, load_budgets, validate_budgets
from arara_quant.utils.data_loading import read_dataframe, read_vector
from arara_quant.utils.yaml_loader import load_yaml_text

DEFAULT_OPTIMIZER_CONFIG = "optimizer_example.yaml"

logger = logging.getLogger(__name__)

__all__ = [
    "OptimizationResult",
    "resolve_config_path",
    "run_optimizer",
]


@dataclass(frozen=True)
class OptimizerConfig:
    config_path: Path
    base_currency: str
    objective: str
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
    long_only: bool = True
    cvar_alpha: float | None = None
    target_return: float | None = None
    max_cvar: float | None = None
    risk_config: Mapping[str, Any] | None = None
    budgets: Sequence[RiskBudget] | None = None
    metaheuristic: Mapping[str, Any] | None = None


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
            candidate = (
                in_configs if in_configs.exists() else settings.project_root / candidate
            )

    candidate = candidate.expanduser().resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"optimizer config not found: {candidate}")
    return candidate


def run_optimizer(
    config_path: str | Path | None = None,
    *,
    dry_run: bool = True,
    settings: Settings | None = None,
    metaheuristic_override: Mapping[str, Any] | None = None,
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
        previous_weights = previous_weights.reindex(
            mu_series.index, fill_value=0.0
        ).astype(float)

    cost_vector = _build_cost_vector(optimizer_config.linear_costs_bps, mu_series.index)

    if optimizer_config.objective.startswith("mean_cvar"):
        cvar_alpha = optimizer_config.cvar_alpha or 0.95
        cvar_config = CvarConfig(
            alpha=cvar_alpha,
            risk_aversion=optimizer_config.risk_aversion,
            long_only=optimizer_config.long_only,
            lower_bounds=lower,
            upper_bounds=upper,
            turnover_penalty=optimizer_config.turnover_penalty,
            turnover_cap=optimizer_config.turnover_cap,
            previous_weights=previous_weights,
            target_return=optimizer_config.target_return,
            max_cvar=optimizer_config.max_cvar,
            solver=optimizer_config.solver,
            solver_kwargs=optimizer_config.solver_kwargs,
        )
        aligned_returns = returns.reindex(columns=mu_series.index)
        cvar_result = solve_cvar_lp(aligned_returns, mu_series, cvar_config)

        result.weights = cvar_result.weights
        result.metrics = {
            "expected_return": cvar_result.expected_return,
            "cvar": cvar_result.cvar,
            "var": cvar_result.var,
        }
        result.turnover = cvar_result.turnover
        result.cost = None
        result.summary = cvar_result.summary
        result.dry_run = False

        if (
            optimizer_config.max_cvar is not None
            and cvar_result.cvar > optimizer_config.max_cvar + 1e-9
        ):
            result.notes.append("CVaR limit reported above configured threshold")
        if optimizer_config.budgets:
            result.notes.append(
                "Group budgets are not yet supported in mean-CVaR solver; skipped."
            )

        return result

    if optimizer_config.budgets:
        validate_budgets(optimizer_config.budgets, mu_series.index)

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
        budgets=optimizer_config.budgets,
    )

    meta_cfg = metaheuristic_override or optimizer_config.metaheuristic
    meta_result: MetaheuristicResult | None = None
    if (
        meta_cfg
        and not dry_run
        and optimizer_config.objective.startswith("mean_variance")
    ):
        mv_config, meta_result = _apply_metaheuristic(
            returns,
            mu_series,
            mv_config,
            optimizer_config,
            meta_cfg,
        )

    mv_result = solve_mean_variance(mu_series, cov_matrix, mv_config)

    metrics = {
        "expected_return": mv_result.expected_return,
        "variance": mv_result.variance,
        "objective_value": mv_result.objective_value,
    }

    if meta_result is not None:
        params = {k: meta_result.params.get(k) for k in ("lambda", "eta", "tau")}
        card = int(meta_result.metrics.get("cardinality", 0))

        lambda_val = params.get("lambda")
        eta_val = params.get("eta")
        tau_val = params.get("tau")

        lambda_disp = float(lambda_val) if lambda_val is not None else float(
            optimizer_config.risk_aversion
        )
        eta_disp = float(eta_val) if eta_val is not None else float(
            optimizer_config.turnover_penalty
        )
        tau_disp = None if tau_val is None else float(tau_val)

        result.notes.append(
            "Metaheuristic tuned λ={:.4g}, η={:.4g}, τ={} (k={})".format(
                lambda_disp,
                eta_disp,
                "None" if tau_disp is None else f"{tau_disp:.4g}",
                card,
            )
        )
        metrics["metaheuristic"] = {
            "fitness": float(meta_result.fitness),
            "cardinality": card,
            "params": {
                "lambda": lambda_disp,
                "eta": eta_disp,
                "tau": tau_disp,
            },
        }

    result.weights = mv_result.weights
    result.metrics = metrics
    result.turnover = mv_result.turnover
    result.cost = mv_result.cost
    result.summary = mv_result.summary
    result.dry_run = False

    if mv_result.summary.status == "infeasible":
        result.notes.append("Solver reported infeasible problem")

    if (
        optimizer_config.turnover_cap is not None
        and optimizer_config.turnover_cap > 0
        and result.turnover is not None
    ):
        target = float(optimizer_config.turnover_cap)
        if result.turnover > target + 1e-6:
            result.notes.append(
                f"Turnover {result.turnover:.2%} exceeded soft target {target:.2%}; consider increasing η."
            )
        else:
            result.notes.append(
                f"Turnover {result.turnover:.2%} within soft target {target:.2%}."
            )

    return result


def _load_config(path: Path, settings: Settings) -> OptimizerConfig:
    raw = load_yaml_text(path.read_text(encoding="utf-8"))

    base_currency = raw.get("base_currency", settings.base_currency)
    optimizer_section = raw.get("optimizer", {})
    risk_limits = raw.get("risk_limits", {})

    objective = str(
        optimizer_section.get("objective", DEFAULT_OPTIMIZER_YAML.objective)
    ).lower()
    long_only = bool(
        optimizer_section.get("long_only", DEFAULT_OPTIMIZER_YAML.long_only)
    )

    risk_aversion = float(
        optimizer_section.get("lambda", DEFAULT_OPTIMIZER_YAML.risk_aversion)
    )
    turnover_penalty = float(
        optimizer_section.get("eta", DEFAULT_OPTIMIZER_YAML.turnover_penalty)
    )
    turnover_cap = optimizer_section.get("tau")
    if turnover_cap is not None:
        turnover_cap = float(turnover_cap)

    target_return = optimizer_section.get("target_return")
    if target_return is not None:
        target_return = float(target_return)

    min_weight = float(
        optimizer_section.get("min_weight", DEFAULT_OPTIMIZER_YAML.min_weight)
    )
    max_weight = float(
        optimizer_section.get("max_weight", DEFAULT_OPTIMIZER_YAML.max_weight)
    )

    linear_costs = (
        raw.get("estimators", {})
        .get("costs", {})
        .get("linear_bps", DEFAULT_OPTIMIZER_YAML.linear_costs_bps)
    )

    mu_config = raw.get("estimators", {}).get("mu", {})
    sigma_config = raw.get("estimators", {}).get("sigma", {})

    def _collect_budgets(raw_budgets: Any) -> list[RiskBudget]:
        collected: list[RiskBudget] = []
        if not raw_budgets:
            return collected
        if isinstance(raw_budgets, RiskBudget):
            return [raw_budgets]
        if not isinstance(raw_budgets, (list, tuple, set, Mapping)):
            raise TypeError(
                "Budgets must be mappings, iterables, or RiskBudget objects."
            )
        iterable = (
            raw_budgets
            if isinstance(raw_budgets, (list, tuple, set))
            else [raw_budgets]
        )
        direct: list[RiskBudget] = []
        loadable: list[Mapping[str, Any]] = []
        for item in iterable:
            if isinstance(item, RiskBudget):
                direct.append(item)
            elif isinstance(item, Mapping):
                loadable.append(item)
            else:
                raise TypeError("Budgets must be mappings or RiskBudget objects.")
        collected.extend(direct)
        if loadable:
            collected.extend(load_budgets(loadable))
        return collected

    risk_section = optimizer_section.get("risk") or optimizer_section.get(
        "risk_constraints"
    )
    risk_config = None
    budgets_accum: list[RiskBudget] = []
    if isinstance(risk_section, Mapping):
        risk_config = {k: v for k, v in risk_section.items()}
        budgets_accum.extend(_collect_budgets(risk_config.pop("budgets", None)))

    portfolio_section = raw.get("portfolio")
    if isinstance(portfolio_section, Mapping):
        portfolio_risk = portfolio_section.get("risk")
        if isinstance(portfolio_risk, Mapping):
            budgets_accum.extend(_collect_budgets(portfolio_risk.get("budgets")))
    budgets: Sequence[RiskBudget] | None = budgets_accum or None

    data_section = raw.get("data", {})
    returns_path_raw = data_section.get("returns") or data_section.get("returns_path")
    if returns_path_raw is None:
        returns_path = (
            settings.processed_data_dir / DEFAULT_OPTIMIZER_YAML.returns_filename
        )
    else:
        returns_path = _resolve_relative_path(
            returns_path_raw, base=path.parent, settings=settings, must_exist=False
        )

    previous_weights: pd.Series | None = None
    state_section = raw.get("state", {})
    prev_data = state_section.get("previous_weights")
    if isinstance(prev_data, str):
        prev_path = _resolve_relative_path(
            prev_data, base=path.parent, settings=settings, must_exist=False
        )
        if prev_path.exists():
            previous_weights = read_vector(prev_path)
    elif isinstance(prev_data, Mapping):
        previous_weights = pd.Series(
            {k: float(v) for k, v in prev_data.items()}, dtype=float
        )

    solver = optimizer_section.get("solver")
    solver_kwargs = optimizer_section.get("solver_kwargs", {})

    metaheuristic_cfg = optimizer_section.get("metaheuristic")
    if isinstance(metaheuristic_cfg, (str, Path)):
        meta_path = _resolve_relative_path(
            Path(metaheuristic_cfg), base=path.parent, settings=settings
        )
        metaheuristic_cfg = load_yaml_text(meta_path.read_text(encoding="utf-8"))
    elif metaheuristic_cfg is not None and not isinstance(metaheuristic_cfg, Mapping):
        raise TypeError("optimizer.metaheuristic must be a mapping or path")

    cvar_alpha = None
    max_cvar = None
    if isinstance(risk_limits, Mapping):
        if "cvar_alpha" in risk_limits:
            cvar_alpha = float(risk_limits["cvar_alpha"])
        if "cvar_max" in risk_limits:
            max_cvar = float(risk_limits["cvar_max"])
        if target_return is None and "target_return" in risk_limits:
            target_return = float(risk_limits["target_return"])

    if "cvar_alpha" in optimizer_section:
        cvar_alpha = float(optimizer_section["cvar_alpha"])
    if "cvar_max" in optimizer_section:
        max_cvar = float(optimizer_section["cvar_max"])

    return OptimizerConfig(
        config_path=path,
        base_currency=base_currency,
        objective=objective,
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
        long_only=long_only,
        cvar_alpha=cvar_alpha,
        target_return=target_return,
        max_cvar=max_cvar,
        risk_config=risk_config,
        budgets=budgets,
        metaheuristic=metaheuristic_cfg,
    )


def _resolve_relative_path(
    value: str | Path, *, base: Path, settings: Settings, must_exist: bool = True
) -> Path:
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


def _estimate_inputs(
    returns: pd.DataFrame, config: OptimizerConfig
) -> tuple[pd.Series, pd.DataFrame]:
    mu_config = config.mu_config
    sigma_config = config.sigma_config

    mu_window = int(
        mu_config.get("window_days", DEFAULT_OPTIMIZER_YAML.estimators_mu.window_days)
        or 0
    )
    sigma_window = int(
        sigma_config.get(
            "window_days", DEFAULT_OPTIMIZER_YAML.estimators_sigma.window_days
        )
        or 0
    )

    data_for_mu = returns.tail(mu_window) if mu_window > 0 else returns
    data_for_sigma = returns.tail(sigma_window) if sigma_window > 0 else returns

    mu_method = (
        mu_config.get("method") or DEFAULT_OPTIMIZER_YAML.estimators_mu.method
    ).lower()
    if mu_method in {"simple", "mean"}:
        mu_series = mean_estimators.mean_return(data_for_mu, method="simple")
    elif mu_method == "geometric":
        mu_series = mean_estimators.mean_return(data_for_mu, method="geometric")
    elif mu_method == "huber":
        delta = float(
            mu_config.get("delta", DEFAULT_OPTIMIZER_YAML.estimators_mu.huber_delta)
        )
        mu_series, _ = mean_estimators.huber_mean(data_for_mu, c=delta)
    elif mu_method in {"shrunk", "shrunk_50", "shrinkage"}:
        strength = float(
            mu_config.get(
                "strength",
                mu_config.get(
                    "shrink_strength",
                    DEFAULT_OPTIMIZER_YAML.estimators_mu.shrink_strength,
                ),
            )
        )
        if not 0.0 <= strength <= 1.0:
            raise ValueError("Shrinkage strength must lie in [0, 1].")
        prior_value = mu_config.get("prior")
        prior: Any
        if prior_value is None:
            prior = DEFAULT_OPTIMIZER_YAML.estimators_mu.prior
        elif isinstance(prior_value, pd.Series):
            prior = prior_value
        elif isinstance(prior_value, Mapping):
            prior = pd.Series(
                {str(k): float(v) for k, v in prior_value.items()}, dtype=float
            )
        else:
            prior = prior_value
        mu_series = mean_estimators.shrunk_mean(
            data_for_mu, strength=strength, prior=prior
        )
    elif mu_method == "student_t":
        nu = float(
            mu_config.get("nu", DEFAULT_OPTIMIZER_YAML.estimators_mu.student_t_nu)
        )
        mu_series = mean_estimators.student_t_mean(data_for_mu, nu=nu)
    else:
        raise ValueError(f"Unsupported mean estimator '{mu_method}'")

    sigma_method = (
        sigma_config.get("method") or DEFAULT_OPTIMIZER_YAML.estimators_sigma.method
    )
    sigma_key = sigma_method.strip().lower().replace("-", "_")
    if sigma_key == "ledoit_wolf":
        nonlinear = bool(
            sigma_config.get(
                "nonlinear", DEFAULT_OPTIMIZER_YAML.estimators_sigma.nonlinear
            )
        )
        if nonlinear:
            cov_matrix = covariance_estimators.nonlinear_shrinkage(data_for_sigma)
        else:
            cov_matrix, _ = covariance_estimators.ledoit_wolf_shrinkage(data_for_sigma)
    elif sigma_key == "nonlinear":
        cov_matrix = covariance_estimators.nonlinear_shrinkage(data_for_sigma)
    elif sigma_key in {"sample", "empirical"}:
        cov_matrix = covariance_estimators.sample_cov(data_for_sigma)
    elif sigma_key == "oas":
        assume_centered = bool(
            sigma_config.get(
                "assume_centered",
                DEFAULT_OPTIMIZER_YAML.estimators_sigma.assume_centered,
            )
        )
        cov_matrix, _ = covariance_estimators.oas_shrinkage(
            data_for_sigma, assume_centered=assume_centered
        )
    elif sigma_key in {"mincovdet", "mcd", "min_cov_det"}:
        assume_centered = bool(
            sigma_config.get(
                "assume_centered",
                DEFAULT_OPTIMIZER_YAML.estimators_sigma.assume_centered,
            )
        )
        cov_matrix = covariance_estimators.min_cov_det(
            data_for_sigma,
            support_fraction=sigma_config.get("support_fraction"),
            assume_centered=assume_centered,
            random_state=sigma_config.get("random_state"),
        )
    elif sigma_key == "tyler":
        cov_matrix = covariance_estimators.tyler_m_estimator(data_for_sigma)
    elif sigma_key in {"graphical_lasso", "glasso"}:
        glasso_defaults = DEFAULT_OPTIMIZER_YAML.estimators_sigma.graphical_lasso
        alpha = float(sigma_config.get("alpha", glasso_defaults.alpha))
        max_iter = int(sigma_config.get("max_iter", glasso_defaults.max_iter))
        tol = float(sigma_config.get("tol", glasso_defaults.tol))
        assume_centered = bool(
            sigma_config.get(
                "assume_centered",
                DEFAULT_OPTIMIZER_YAML.estimators_sigma.assume_centered,
            )
        )
        enet_tol = float(sigma_config.get("enet_tol", glasso_defaults.enet_tol))
        mode = str(sigma_config.get("mode", glasso_defaults.mode))
        sparsity_tol = float(
            sigma_config.get("sparsity_tol", glasso_defaults.sparsity_tol)
        )
        cov_matrix, precision_matrix = covariance_estimators.graphical_lasso_cov(
            data_for_sigma,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            assume_centered=assume_centered,
            enet_tol=enet_tol,
            mode=mode,
        )
        if precision_matrix.shape[0] > 1:
            mask = ~np.eye(precision_matrix.shape[0], dtype=bool)
            off_diag = np.abs(precision_matrix.to_numpy()[mask])
            if off_diag.size > 0:
                sparse_ratio = float(np.mean(off_diag < sparsity_tol))
            logger.info(
                "Graphical Lasso alpha=%.4f produced %.1f%% sparse precision "
                "(tol=%g, mode=%s).",
                alpha,
                100.0 * sparse_ratio,
                sparsity_tol,
                mode,
            )
    else:
        raise ValueError(f"Unsupported covariance estimator '{sigma_method}'")

    mu_series = mu_series.astype(float)
    cov_matrix = cov_matrix.astype(float)
    cov_matrix = cov_matrix.reindex(
        index=mu_series.index, columns=mu_series.index
    ).fillna(0.0)
    cov_matrix = covariance_estimators.project_to_psd(cov_matrix, epsilon=1e-9)

    return mu_series, cov_matrix


def _clone_mv_config(
    config: MeanVarianceConfig, **updates: Any
) -> MeanVarianceConfig:
    payload = {field.name: getattr(config, field.name) for field in fields(MeanVarianceConfig)}
    payload.update(updates)
    return MeanVarianceConfig(**payload)


def _parse_float_pair(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    raise ValueError(
        "turnover_target must be a sequence with exactly two numeric entries"
    )


def _parse_int_pair(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(
        "cardinality_target must be a sequence with exactly two integers"
    )


def _apply_metaheuristic(
    returns: pd.DataFrame,
    mu_series: pd.Series,
    base_config: MeanVarianceConfig,
    optimizer_config: OptimizerConfig,
    meta_cfg: Mapping[str, Any],
) -> tuple[MeanVarianceConfig, MetaheuristicResult | None]:
    if not isinstance(meta_cfg, Mapping):
        raise TypeError("metaheuristic configuration must be a mapping")
    if not meta_cfg.get("enabled", True):
        return base_config, None

    ga_cfg = meta_cfg.get("ga")
    if not isinstance(ga_cfg, Mapping):
        raise ValueError("metaheuristic config must include a 'ga' mapping")

    window_days = int(meta_cfg.get("window_days", 0) or 0)
    calibration_returns = returns.tail(window_days) if window_days > 0 else returns
    mu_meta, cov_meta = _estimate_inputs(calibration_returns, optimizer_config)

    ga_cfg_local = dict(ga_cfg)
    if "parallel" not in ga_cfg_local and meta_cfg.get("parallel"):
        ga_cfg_local["parallel"] = dict(meta_cfg["parallel"])

    turnover_target = _parse_float_pair(meta_cfg.get("turnover_target"))
    cardinality_target = _parse_int_pair(meta_cfg.get("cardinality_target"))
    penalty_weights = dict(meta_cfg.get("penalty_weights", {}))

    meta_result = metaheuristic_outer(
        mu_meta,
        cov_meta,
        base_config,
        ga_config=ga_cfg_local,
        turnover_target=turnover_target,
        cardinality_target=cardinality_target,
        penalty_weights=penalty_weights,
    )

    updates: dict[str, Any] = {}
    params = meta_result.params or {}
    if params.get("lambda") is not None:
        updates["risk_aversion"] = float(params["lambda"])
    if params.get("eta") is not None:
        updates["turnover_penalty"] = float(params["eta"])
    if "tau" in params:
        tau_value = params.get("tau")
        updates["turnover_cap"] = (
            None if tau_value is None else float(tau_value)
        )

    apply_selection = bool(meta_cfg.get("apply_selection", True))
    selected_assets = [
        asset for asset in meta_result.selected_assets if asset in mu_series.index
    ]
    if apply_selection and selected_assets and len(selected_assets) < len(mu_series.index):
        lower = base_config.lower_bounds.reindex(mu_series.index).fillna(0.0).copy()
        upper = base_config.upper_bounds.reindex(mu_series.index).fillna(1.0).copy()
        inactive = ~lower.index.isin(selected_assets)
        lower.loc[inactive] = 0.0
        upper.loc[inactive] = 0.0
        updates["lower_bounds"] = lower
        updates["upper_bounds"] = upper

    tuned_config = _clone_mv_config(base_config, **updates) if updates else base_config
    return tuned_config, meta_result


def _build_cost_vector(
    costs: float | Mapping[str, float], index: pd.Index
) -> pd.Series | None:
    if isinstance(costs, Mapping):
        series = pd.Series({str(k): float(v) for k, v in costs.items()}, dtype=float)
        return (
            series.reindex(index).fillna(series.mean() if not series.empty else 0.0)
            / 10_000.0
        )
    scalar = float(costs)
    if scalar == 0:
        return None
    return pd.Series(scalar / 10_000.0, index=index, dtype=float)
