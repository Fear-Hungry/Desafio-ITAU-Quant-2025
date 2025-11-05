"""Walk-forward out-of-sample evaluation utilities for baseline strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from itau_quant.estimators.cov import ledoit_wolf_shrinkage
from itau_quant.estimators.mu import shrunk_mean
from itau_quant.optimization.core.mv_qp import MeanVarianceConfig, solve_mean_variance
from itau_quant.optimization.core.risk_parity import risk_parity
from itau_quant.optimization.heuristics.hrp import hierarchical_risk_parity

__all__ = [
    "StrategySpec",
    "OOSResult",
    "compare_baselines",
    "default_strategies",
    "stress_test",
]


@dataclass(frozen=True)
class StrategySpec:
    """Description of a candidate strategy for the OOS evaluation."""

    name: str
    builder: Callable[[pd.DataFrame, pd.Series | None], pd.Series]


@dataclass(frozen=True)
class OOSResult:
    """Container with walk-forward returns, metrics, and weight history."""

    returns: pd.DataFrame
    metrics: pd.DataFrame
    weights: dict[str, list[tuple[pd.Timestamp, pd.Series]]] = field(
        default_factory=dict
    )
    turnovers: dict[str, list[float]] = field(default_factory=dict)


def _ensure_frame(data: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(data, pd.Series):
        return data.to_frame()
    return data.copy()


def _max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    cumulative = (1.0 + series).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative / peak) - 1.0
    return float(drawdown.min())


def _cvar(series: pd.Series, alpha: float = 0.95) -> float:
    if series.empty:
        return 0.0
    tail = np.ceil((1.0 - alpha) * len(series))
    tail = int(max(tail, 1))
    sorted_returns = np.sort(series.to_numpy())
    return float(sorted_returns[:tail].mean())


def _sharpe_ratio(daily_returns: pd.Series) -> float:
    mean = float(daily_returns.mean())
    std = float(daily_returns.std(ddof=1))
    if std <= 0:
        return 0.0
    return float(mean / std * np.sqrt(252.0))


def _moving_block_bootstrap(
    values: np.ndarray,
    *,
    n_bootstrap: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_obs = len(values)
    if block_size <= 1 or block_size >= n_obs:
        idx = rng.integers(0, n_obs, size=(n_bootstrap, n_obs))
        return values[idx]

    int(np.ceil(n_obs / block_size))
    samples = np.empty((n_bootstrap, n_obs), dtype=float)
    for b in range(n_bootstrap):
        pos = 0
        while pos < n_obs:
            start = rng.integers(0, n_obs - block_size + 1)
            end = min(start + block_size, n_obs)
            length = end - start
            samples[b, pos : pos + length] = values[start:end]
            pos += length
    return samples


def _bootstrap_sharpe_ci(
    daily_returns: pd.Series,
    *,
    n_bootstrap: int,
    confidence: float,
    block_size: int | None,
    random_state: int | None,
) -> tuple[float, float]:
    if daily_returns.empty or n_bootstrap <= 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(random_state)
    data = daily_returns.to_numpy(dtype=float)
    samples = (
        _moving_block_bootstrap(
            data, n_bootstrap=n_bootstrap, block_size=block_size or 1, rng=rng
        )
        if block_size
        else data[rng.integers(0, len(data), size=(n_bootstrap, len(data)))]
    )
    means = samples.mean(axis=1)
    stds = samples.std(axis=1, ddof=1)
    valid = stds > 0
    if not valid.any():
        return float("nan"), float("nan")
    sharpes = means[valid] / stds[valid] * np.sqrt(252.0)
    alpha = (1.0 - confidence) / 2.0
    low = float(np.quantile(sharpes, alpha))
    high = float(np.quantile(sharpes, 1.0 - alpha))
    return low, high


def _compute_metrics(
    daily_returns: pd.Series,
    *,
    avg_turnover: float,
    total_cost: float,
    bootstrap_iterations: int | None,
    confidence: float,
    block_size: int | None,
    random_state: int | None,
) -> dict[str, float]:
    if daily_returns.empty:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "volatility": 0.0,
            "sharpe": 0.0,
            "cvar_95": 0.0,
            "max_drawdown": 0.0,
            "avg_turnover": 0.0,
            "total_cost": 0.0,
            "sharpe_ci_low": float("nan"),
            "sharpe_ci_high": float("nan"),
        }

    periods = len(daily_returns)
    cumulative = (1.0 + daily_returns).prod()
    total_return = cumulative - 1.0

    mean = float(daily_returns.mean())
    std = float(daily_returns.std(ddof=1))
    ann_factor = np.sqrt(252.0)

    sharpe = (mean / std * ann_factor) if std > 0 else 0.0
    volatility = std * ann_factor if std > 0 else 0.0
    annualized = (1.0 + total_return) ** (252.0 / periods) - 1.0 if periods > 0 else 0.0

    cvar_daily = float(_cvar(daily_returns, alpha=0.95))
    cvar_annual = cvar_daily * ann_factor  # √252 scaling

    metrics = {
        "total_return": float(total_return),
        "annualized_return": float(annualized),
        "volatility": float(volatility),
        "sharpe": float(sharpe),
        "cvar_95": cvar_daily,  # Daily CVaR for monitoring
        "cvar_95_annual": cvar_annual,  # Annualized CVaR for targets
        "max_drawdown": float(_max_drawdown(daily_returns)),
        "avg_turnover": float(avg_turnover),
        "total_cost": float(total_cost),
    }

    if bootstrap_iterations:
        ci_low, ci_high = _bootstrap_sharpe_ci(
            daily_returns,
            n_bootstrap=bootstrap_iterations,
            confidence=confidence,
            block_size=block_size,
            random_state=random_state,
        )
        metrics["sharpe_ci_low"] = ci_low
        metrics["sharpe_ci_high"] = ci_high
    else:
        metrics["sharpe_ci_low"] = float("nan")
        metrics["sharpe_ci_high"] = float("nan")

    return metrics


def default_strategies(
    *,
    max_position: float = 0.10,
    risk_aversion: float = 4.0,
    shrink_strength: float = 0.5,
) -> list[StrategySpec]:
    """Return the default baseline strategies used in the PRD."""

    def equal_weight(train_returns: pd.DataFrame, _: pd.Series | None) -> pd.Series:
        columns = train_returns.columns
        weights = pd.Series(1.0 / len(columns), index=columns, dtype=float)
        return weights

    def min_variance(train_returns: pd.DataFrame, _: pd.Series | None) -> pd.Series:
        cov_daily, _ = ledoit_wolf_shrinkage(train_returns)
        cov = cov_daily * 252.0
        assets = cov.columns
        mu_zero = pd.Series(0.0, index=assets, dtype=float)
        bounds = pd.Series(max_position, index=assets, dtype=float)
        config = MeanVarianceConfig(
            risk_aversion=1e6,
            turnover_penalty=0.0,
            turnover_cap=None,
            lower_bounds=pd.Series(0.0, index=assets, dtype=float),
            upper_bounds=bounds,
            previous_weights=pd.Series(0.0, index=assets, dtype=float),
            cost_vector=None,
            solver="CLARABEL",
        )
        result = solve_mean_variance(mu_zero, cov, config)
        return result.weights.reindex(assets).fillna(0.0)

    def risk_parity_erc(train_returns: pd.DataFrame, _: pd.Series | None) -> pd.Series:
        cov_daily, _ = ledoit_wolf_shrinkage(train_returns)
        cov = cov_daily * 252.0
        try:
            result = risk_parity(cov, config={"method": "log_barrier"})
            weights = result.weights.clip(lower=0.0, upper=max_position)
            if weights.sum() == 0:
                return equal_weight(train_returns, None)
            return weights / weights.sum()
        except Exception:
            return equal_weight(train_returns, None)

    def hrp_strategy(train_returns: pd.DataFrame, _: pd.Series | None) -> pd.Series:
        cov_daily, _ = ledoit_wolf_shrinkage(train_returns)
        cov = cov_daily * 252.0
        try:
            weights = hierarchical_risk_parity(cov)
            weights = weights.clip(lower=0.0, upper=max_position)
            if weights.sum() == 0:
                return equal_weight(train_returns, None)
            return weights / weights.sum()
        except Exception:
            return equal_weight(train_returns, None)

    def shrunk_mv(train_returns: pd.DataFrame, _: pd.Series | None) -> pd.Series:
        assets = train_returns.columns
        mu_daily = shrunk_mean(train_returns, strength=shrink_strength, prior=0.0)
        mu = mu_daily * 252.0
        cov_daily, _ = ledoit_wolf_shrinkage(train_returns)
        cov = cov_daily * 252.0
        prev = pd.Series(0.0, index=assets, dtype=float)
        bounds = pd.Series(max_position, index=assets, dtype=float)
        config = MeanVarianceConfig(
            risk_aversion=risk_aversion,
            turnover_penalty=0.0,
            turnover_cap=None,
            lower_bounds=pd.Series(0.0, index=assets, dtype=float),
            upper_bounds=bounds,
            previous_weights=prev,
            cost_vector=None,
            solver="CLARABEL",
        )
        result = solve_mean_variance(mu, cov, config)
        weights = result.weights.reindex(assets).fillna(0.0)
        if weights.sum() == 0:
            return equal_weight(train_returns, None)
        return weights / weights.sum()

    def sixty_forty(train_returns: pd.DataFrame, _: pd.Series | None) -> pd.Series:
        columns = train_returns.columns
        weights = pd.Series(0.0, index=columns, dtype=float)
        if "SPY" in columns and "IEF" in columns:
            weights["SPY"] = 0.60
            weights["IEF"] = 0.40
        else:
            equities = [
                c for c in columns if c.startswith(("SP", "QQQ", "IWM", "VTV", "VUG"))
            ]
            bonds = [
                c for c in columns if c.startswith(("IE", "TLT", "SHY", "LQD", "HYG"))
            ]
            if equities and bonds:
                weights.loc[equities] = 0.60 / len(equities)
                weights.loc[bonds] = 0.40 / len(bonds)
            else:
                return equal_weight(train_returns, None)
        weights = weights.clip(0.0, max_position)
        if weights.sum() == 0:
            return equal_weight(train_returns, None)
        return weights / weights.sum()

    return [
        StrategySpec("equal_weight", equal_weight),
        StrategySpec("min_variance_lw", min_variance),
        StrategySpec("risk_parity", risk_parity_erc),
        StrategySpec("hrp", hrp_strategy),
        StrategySpec("shrunk_mv", shrunk_mv),
        StrategySpec("sixty_forty", sixty_forty),
    ]


def compare_baselines(
    returns: pd.DataFrame,
    *,
    strategies: Sequence[StrategySpec] | None = None,
    train_window: int = 252,
    test_window: int = 21,
    purge_window: int = 5,
    embargo_window: int = 5,
    costs_bps: float = 10.0,
    max_position: float = 0.10,
    bootstrap_iterations: int | None = None,
    confidence: float = 0.95,
    block_size: int | None = None,
    random_state: int | None = None,
) -> OOSResult:
    """Run a walk-forward evaluation comparing multiple strategies."""

    if train_window <= 0 or test_window <= 0:
        raise ValueError("train_window and test_window must be positive.")

    if strategies is None:
        strategies = default_strategies(max_position=max_position)

    returns = _ensure_frame(returns).astype(float).dropna(how="all")
    if returns.empty:
        raise ValueError("returns must not be empty.")

    results: dict[str, list[pd.Series]] = {spec.name: [] for spec in strategies}
    turnovers: dict[str, list[float]] = {spec.name: [] for spec in strategies}
    costs: dict[str, list[float]] = {spec.name: [] for spec in strategies}
    weights_log: dict[str, list[tuple[pd.Timestamp, pd.Series]]] = {
        spec.name: [] for spec in strategies
    }
    prev_weights: dict[str, pd.Series] = {
        spec.name: pd.Series(0.0, index=returns.columns, dtype=float)
        for spec in strategies
    }

    test_start_idx = train_window + purge_window

    while test_start_idx + test_window <= len(returns):
        train_start_idx = test_start_idx - purge_window - train_window
        if train_start_idx < 0:
            break

        train_slice = returns.iloc[train_start_idx : test_start_idx - purge_window]
        test_slice = returns.iloc[test_start_idx : test_start_idx + test_window]
        if train_slice.empty or test_slice.empty:
            break

        rebalance_date = test_slice.index[0]

        for spec in strategies:
            prev = prev_weights[spec.name]
            weights = (
                spec.builder(train_slice, prev).reindex(returns.columns).fillna(0.0)
            )
            weights = weights.clip(lower=0.0, upper=max_position)
            total = float(weights.sum())
            if total > 0:
                weights = weights / total
            turnover = float(np.abs(weights - prev).sum())
            transaction_cost = turnover * (costs_bps / 10_000.0)

            strategy_returns = test_slice.mul(weights, axis=1).sum(axis=1)
            if not strategy_returns.empty:
                strategy_returns.iloc[0] -= transaction_cost

            results[spec.name].append(strategy_returns)
            turnovers[spec.name].append(turnover)
            costs[spec.name].append(transaction_cost)
            weights_log[spec.name].append((rebalance_date, weights))
            prev_weights[spec.name] = weights

        test_start_idx += test_window + embargo_window

    combined_returns: dict[str, pd.Series] = {}
    metrics_records: list[dict[str, float | str]] = []

    for spec in strategies:
        name = spec.name
        if results[name]:
            joined = pd.concat(results[name]).sort_index()
        else:
            joined = pd.Series(dtype=float)
        combined_returns[name] = joined
        avg_turnover = float(np.mean(turnovers[name])) if turnovers[name] else 0.0
        total_cost = float(np.sum(costs[name])) if costs[name] else 0.0
        metrics = _compute_metrics(
            joined,
            avg_turnover=avg_turnover,
            total_cost=total_cost,
            bootstrap_iterations=bootstrap_iterations,
            confidence=confidence,
            block_size=block_size,
            random_state=random_state,
        )
        metrics["strategy"] = name
        metrics_records.append(metrics)

    returns_df = pd.DataFrame(combined_returns).fillna(0.0)
    metrics_df = pd.DataFrame(metrics_records).set_index("strategy")

    return OOSResult(
        returns=returns_df,
        metrics=metrics_df,
        weights=weights_log,
        turnovers=turnovers,
    )


def stress_test(
    oos_returns: pd.DataFrame,
    periods: Mapping[str, tuple[pd.Timestamp | str, pd.Timestamp | str]],
    *,
    bootstrap_iterations: int | None = None,
    confidence: float = 0.95,
    block_size: int | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Evaluate strategy performance during specified stress periods."""

    records: list[dict[str, float | str]] = []
    returns = _ensure_frame(oos_returns).astype(float)

    for label, (start, end) in periods.items():
        window = returns.loc[str(start) : str(end)]
        if window.empty:
            continue
        for strategy in window.columns:
            daily = window[strategy]
            metrics = _compute_metrics(
                daily,
                avg_turnover=0.0,
                total_cost=0.0,
                bootstrap_iterations=bootstrap_iterations,
                confidence=confidence,
                block_size=block_size,
                random_state=random_state,
            )
            records.append(
                {
                    "period": label,
                    "strategy": strategy,
                    "total_return": metrics["total_return"],
                    "annualized_return": metrics["annualized_return"],
                    "volatility": metrics["volatility"],
                    "sharpe": metrics["sharpe"],
                    "cvar_95": metrics["cvar_95"],
                    "max_drawdown": metrics["max_drawdown"],
                }
            )

    if not records:
        return pd.DataFrame(columns=["period", "strategy"])

    return pd.DataFrame(records)
