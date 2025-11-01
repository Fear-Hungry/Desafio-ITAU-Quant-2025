"""Backtesting engine and companion utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from itau_quant.backtesting.bookkeeping import PortfolioLedger, build_ledger
from itau_quant.backtesting.metrics import PortfolioMetrics, compute_performance_metrics
from itau_quant.backtesting.risk_monitor import evaluate_turnover_band
from itau_quant.backtesting.walk_forward import generate_walk_forward_splits
from itau_quant.config import Settings, get_settings
from itau_quant.evaluation.walkforward_report import (
    WalkForwardSummary,
    compute_wf_summary_stats,
)
from itau_quant.optimization.solvers import resolve_config_path
from itau_quant.portfolio import MarketData, rebalance
from itau_quant.utils.data_loading import read_dataframe
from itau_quant.utils.yaml_loader import load_yaml_text

DEFAULT_BACKTEST_CONFIG = "optimizer_example.yaml"

__all__ = ["BacktestConfig", "BacktestResult", "run_backtest"]


@dataclass(frozen=True)
class BacktestConfig:
    config_path: Path
    base_currency: str
    turnover_target: tuple[float, float] | None
    portfolio_config: Mapping[str, Any]
    capital: float
    walkforward: Mapping[str, int]
    returns_path: Path
    prices_path: Path | None
    risk_free_path: Path | None = None
    evaluation_horizons: tuple[int, ...] = (21, 63, 126)


@dataclass(slots=True)
class BacktestResult:
    config_path: Path
    environment: str
    base_currency: str
    dry_run: bool
    metrics: PortfolioMetrics | None = None
    ledger: PortfolioLedger | None = None
    weights: pd.DataFrame | None = None
    trades: pd.DataFrame | None = None
    split_metrics: pd.DataFrame | None = None
    horizon_metrics: pd.DataFrame | None = None
    walkforward_summary: WalkForwardSummary | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self, include_timeseries: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "config_path": str(self.config_path),
            "environment": self.environment,
            "base_currency": self.base_currency,
            "dry_run": self.dry_run,
            "status": "preview" if self.dry_run else "completed",
        }
        if self.metrics is not None:
            payload["metrics"] = self.metrics.as_dict()
        if self.notes:
            payload["notes"] = list(self.notes)
        if include_timeseries and self.ledger is not None:
            payload["ledger"] = self.ledger.as_dict()
        if include_timeseries and self.weights is not None:
            weights_df = self.weights.reset_index()
            if "date" in weights_df.columns:
                weights_df["date"] = pd.to_datetime(weights_df["date"]).dt.strftime(
                    "%Y-%m-%d"
                )
            payload["weights"] = weights_df.to_dict(orient="records")
        if include_timeseries and self.trades is not None:
            payload["trades"] = self.trades.to_dict(orient="records")
        if include_timeseries and self.split_metrics is not None:
            payload["walkforward"] = self.split_metrics.to_dict(orient="records")
        if include_timeseries and self.horizon_metrics is not None:
            payload["horizon_metrics"] = self.horizon_metrics.to_dict(orient="records")
        if self.walkforward_summary is not None:
            payload["walkforward_summary"] = {
                "n_windows": self.walkforward_summary.n_windows,
                "success_rate": self.walkforward_summary.success_rate,
                "avg_sharpe": self.walkforward_summary.avg_sharpe,
                "avg_return": self.walkforward_summary.avg_return,
                "avg_volatility": self.walkforward_summary.avg_volatility,
                "avg_drawdown": self.walkforward_summary.avg_drawdown,
                "avg_turnover": self.walkforward_summary.avg_turnover,
                "avg_cost": self.walkforward_summary.avg_cost,
                "consistency_r2": self.walkforward_summary.consistency_r2,
                "best_window_nav": self.walkforward_summary.best_window_nav,
                "worst_window_nav": self.walkforward_summary.worst_window_nav,
                "range_ratio": self.walkforward_summary.range_ratio,
            }
        return payload


def _load_config(path: Path, settings: Settings) -> BacktestConfig:
    with path.open(encoding="utf-8") as handle:
        raw = load_yaml_text(handle.read())

    base_currency = raw.get("base_currency", settings.base_currency)

    turnover_band = raw.get("rebalancing", {}).get("turnover_target")
    if turnover_band is not None:
        turnover_band = tuple(float(x) for x in turnover_band)

    optimizer_section = raw.get("optimizer", {})
    if not isinstance(optimizer_section, Mapping):
        optimizer_section = {}

    risk_aversion = float(optimizer_section.get("lambda", 5.0))
    turnover_penalty = float(optimizer_section.get("eta", 0.0))
    turnover_cap = optimizer_section.get("tau")
    if turnover_cap is not None:
        turnover_cap = float(turnover_cap)
    elif turnover_band is not None:
        turnover_cap = float(turnover_band[1])

    min_weight = float(optimizer_section.get("min_weight", 0.0))
    max_weight = float(optimizer_section.get("max_weight", 1.0))

    estimators_section = raw.get("estimators", {})
    if not isinstance(estimators_section, Mapping):
        estimators_section = {}
    costs_section = (
        estimators_section.get("costs", {})
        if isinstance(estimators_section.get("costs", {}), Mapping)
        else estimators_section.get("costs", 0.0)
    )
    if isinstance(costs_section, Mapping):
        linear_costs = costs_section.get("linear_bps", 0.0)
    else:
        linear_costs = costs_section or 0.0

    (
        estimators_section.get("mu", {})
        if isinstance(estimators_section.get("mu", {}), Mapping)
        else {}
    )
    (
        estimators_section.get("sigma", {})
        if isinstance(estimators_section.get("sigma", {}), Mapping)
        else {}
    )

    walkforward = raw.get("walkforward", {})
    wf_defaults = {
        "train_days": 252,
        "test_days": 21,
        "purge_days": 0,
        "embargo_days": 0,
    }
    wf_config = dict(wf_defaults)
    for key, value in walkforward.items():
        if key == "evaluation_horizons":
            continue
        if key in wf_defaults:
            try:
                wf_config[key] = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid walkforward parameter '{key}': {value!r}"
                ) from exc
        else:
            wf_config[key] = value
    horizons = walkforward.get("evaluation_horizons") or raw.get("evaluation_horizons")
    if horizons is None:
        evaluation_horizons = (21, 63, 126)
    else:
        if isinstance(horizons, (list, tuple, set)):
            evaluation_horizons = tuple(int(x) for x in horizons if int(x) > 0)
        else:
            evaluation_horizons = (int(horizons),)
        if not evaluation_horizons:
            evaluation_horizons = (21, 63, 126)

    data_section = raw.get("data", {})
    if not isinstance(data_section, Mapping):
        data_section = {}
    returns_entry = data_section.get("returns") or data_section.get("returns_path")
    if returns_entry is None:
        default_returns = settings.data_dir / "processed" / "returns_arara.parquet"
        if not default_returns.exists():
            raise FileNotFoundError(
                "Could not locate returns data. Provide data.returns in the backtest config."
            )
        returns_path = default_returns
    else:
        returns_path = _resolve_data_path(
            returns_entry, base=path.parent, settings=settings
        )

    risk_free_entry = data_section.get("risk_free")
    risk_free_path = None
    if risk_free_entry is not None:
        risk_free_path = _resolve_data_path(
            risk_free_entry, base=path.parent, settings=settings
        )
    prices_entry = data_section.get("prices")
    prices_path = None
    if prices_entry is not None:
        prices_path = _resolve_data_path(
            prices_entry, base=path.parent, settings=settings
        )

    portfolio_section = raw.get("portfolio", {})
    if not isinstance(portfolio_section, Mapping):
        portfolio_section = {}
    capital = float(portfolio_section.get("capital", 1.0))
    portfolio_rounding = portfolio_section.get("rounding", {})
    portfolio_costs = portfolio_section.get("costs", {}) or {}
    if linear_costs and "linear_bps" not in portfolio_costs:
        portfolio_costs.setdefault("linear_bps", linear_costs)
    returns_window_portfolio = int(
        portfolio_section.get("returns_window", wf_defaults["train_days"])
    )
    baseline_config = portfolio_section.get("baseline")

    optimizer_config = {
        "risk_aversion": risk_aversion,
        "turnover_penalty": turnover_penalty,
        "turnover_cap": turnover_cap,
        "min_weight": min_weight,
        "max_weight": max_weight,
        "solver": optimizer_section.get("solver"),
        "solver_kwargs": optimizer_section.get("solver_kwargs"),
    }

    risk_section: dict[str, Any] = {}
    optimizer_risk = optimizer_section.get("risk") or optimizer_section.get(
        "risk_constraints"
    )
    if isinstance(optimizer_risk, Mapping):
        risk_section.update({k: v for k, v in optimizer_risk.items()})

    portfolio_risk = portfolio_section.get("risk") or portfolio_section.get(
        "risk_constraints"
    )
    if isinstance(portfolio_risk, Mapping):
        risk_section.update({k: v for k, v in portfolio_risk.items()})

    # Support shorthand keys declared directly under `portfolio`
    for key in (
        "budgets",
        "max_leverage",
        "factor_exposure",
        "tracking_error",
        "turnover",
    ):
        if key in portfolio_section:
            risk_section[key] = portfolio_section[key]

    portfolio_config: dict[str, Any] = {
        "optimizer": optimizer_config,
        "rounding": portfolio_rounding,
        "costs": portfolio_costs,
        "returns_window": returns_window_portfolio,
    }
    if risk_section:
        portfolio_config["risk"] = risk_section
    if baseline_config:
        portfolio_config["baseline"] = baseline_config

    return BacktestConfig(
        config_path=path,
        base_currency=base_currency,
        turnover_target=turnover_band,
        portfolio_config=portfolio_config,
        capital=capital,
        walkforward=wf_config,
        returns_path=returns_path,
        prices_path=prices_path,
        risk_free_path=risk_free_path,
        evaluation_horizons=evaluation_horizons,
    )


def _resolve_data_path(value: str | Path, *, base: Path, settings: Settings) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (base / candidate).resolve()
    if not candidate.exists():
        alt = (settings.project_root / Path(value)).resolve()
        if alt.exists():
            candidate = alt
    if not candidate.exists():
        raise FileNotFoundError(f"Data file not found: {candidate}")
    return candidate


def _load_market_data(
    config: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None]:
    returns = read_dataframe(config.returns_path)
    if not isinstance(returns.index, pd.DatetimeIndex):
        returns.index = pd.to_datetime(returns.index)
    returns = returns.sort_index()

    if config.prices_path is not None:
        prices = read_dataframe(config.prices_path)
        if isinstance(prices, pd.Series):
            prices = prices.to_frame().T
        if not isinstance(prices.index, pd.DatetimeIndex):
            prices.index = pd.to_datetime(prices.index)
        prices = (
            prices.sort_index().reindex(columns=returns.columns).fillna(method="ffill")
        )
    else:
        prices = (1.0 + returns).cumprod()

    rf = None
    if config.risk_free_path is not None:
        rf_raw = read_dataframe(config.risk_free_path)
        rf = rf_raw if isinstance(rf_raw, pd.Series) else rf_raw.iloc[:, 0]
        if isinstance(rf, pd.DataFrame):
            rf = rf.iloc[:, 0]
        if not isinstance(rf.index, pd.DatetimeIndex):
            rf.index = pd.to_datetime(rf.index)
        rf = rf.sort_index()
    return returns, prices, rf


def _run_simulation(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    rf: pd.Series | None,
    config: BacktestConfig,
) -> tuple[PortfolioLedger, pd.DataFrame, pd.DataFrame, PortfolioMetrics, list[str]]:
    wf = config.walkforward
    splits = list(
        generate_walk_forward_splits(
            returns.index,
            train_window=wf["train_days"],
            test_window=wf["test_days"],
            purge_window=wf["purge_days"],
            embargo_window=wf["embargo_days"],
        )
    )
    if not splits:
        raise ValueError("Not enough data to generate walk-forward splits")

    gross = pd.Series(0.0, index=returns.index, name="gross_return")
    net = pd.Series(0.0, index=returns.index, name="net_return")
    costs = pd.Series(0.0, index=returns.index, name="costs")
    turnover_series = pd.Series(0.0, index=returns.index, name="turnover")

    weight_records: list[pd.Series] = []
    trade_records: list[dict[str, Any]] = []
    split_records: list[dict[str, Any]] = []
    notes: list[str] = []

    previous_weights = pd.Series(0.0, index=returns.columns, dtype=float)
    market = MarketData(prices=prices, returns=returns)

    for split in splits:
        train_slice = split.train_index
        test_slice = split.test_index
        train_returns = returns.loc[train_slice]
        test_returns = returns.loc[test_slice]
        if train_returns.empty or test_returns.empty:
            continue

        first_day = test_slice[0]

        rebal_config = dict(config.portfolio_config)
        rebal_config.setdefault(
            "returns_window",
            config.portfolio_config.get(
                "returns_window", config.walkforward["train_days"]
            ),
        )
        rebalance_result = rebalance(
            date=first_day,
            market_data=market,
            previous_weights=previous_weights,
            capital=config.capital,
            config=rebal_config,
        )

        executed_weights = rebalance_result.rounded_weights.reindex(
            returns.columns, fill_value=0.0
        ).astype(float)
        optimizer_cost = rebalance_result.metrics.optimizer_cost
        rounding_cost = rebalance_result.metrics.rounding_cost
        total_cost = float(optimizer_cost + rounding_cost)
        cost_fraction = total_cost / float(config.capital)
        turnover_value = rebalance_result.metrics.realized_turnover
        turnover_status = evaluate_turnover_band(turnover_value, config.turnover_target)
        trade_records.append(
            {
                "date": first_day,
                "turnover": turnover_value,
                "cost": cost_fraction,
                "optimizer_cost": optimizer_cost,
                "rounding_cost": rounding_cost,
                "status": turnover_status,
                "solver_status": rebalance_result.log.get("solver", {}).get("status"),
                "allocator": rebalance_result.allocator,
                "weights_pre_rounding": rebalance_result.log.get(
                    "weights_pre_rounding"
                ),
                "heuristic_method": rebalance_result.log.get("heuristic", {}).get(
                    "method"
                ),
                "heuristic_diagnostics": rebalance_result.log.get("heuristic", {}).get(
                    "diagnostics"
                ),
            }
        )

        if rebalance_result.log.get("solver", {}).get("status") not in {
            None,
            "optimal",
            "OPTIMAL",
        }:
            notes.append(
                f"Optimizer returned status {rebalance_result.log.get('solver', {}).get('status')} on {first_day.date()}"
            )
        if turnover_status == "above":
            notes.append(f"Turnover above target band on {first_day.date()}")

        if rebalance_result.notes:
            notes.extend(str(note) for note in rebalance_result.notes)

        weight_records.append(pd.Series(executed_weights, name=first_day))

        for idx, date in enumerate(test_slice):
            asset_returns = returns.loc[date].fillna(0.0)
            gross_return = float((executed_weights * asset_returns).sum())
            gross.loc[date] = gross_return
            if idx == 0:
                net_return = gross_return - cost_fraction
                costs.loc[date] = cost_fraction
                turnover_series.loc[date] = turnover_value
            else:
                net_return = gross_return
            net.loc[date] = net_return

        previous_weights = executed_weights

        test_metrics = compute_performance_metrics(
            net.loc[test_slice],
            risk_free=None if rf is None else rf.reindex(test_slice),
        )
        split_records.append(
            {
                "train_start": train_slice[0].strftime("%Y-%m-%d"),
                "train_end": train_slice[-1].strftime("%Y-%m-%d"),
                "test_start": test_slice[0].strftime("%Y-%m-%d"),
                "test_end": test_slice[-1].strftime("%Y-%m-%d"),
                "total_return": test_metrics.total_return,
                "annualized_return": test_metrics.annualized_return,
                "annualized_volatility": test_metrics.annualized_volatility,
                "sharpe_ratio": test_metrics.sharpe_ratio,
                "max_drawdown": test_metrics.max_drawdown,
                "cumulative_nav": test_metrics.cumulative_nav,
                "turnover": turnover_value,
                "cost_fraction": cost_fraction,
            }
        )

    ledger = build_ledger(
        dates=returns.index,
        gross_returns=gross,
        net_returns=net,
        costs=costs,
        turnover=turnover_series,
    )

    weights_df = pd.DataFrame(weight_records).sort_index()
    if not weights_df.empty:
        weights_df.index.name = "date"
    trades_df = pd.DataFrame(trade_records)
    if not trades_df.empty:
        trades_df.sort_values("date", inplace=True)
        trades_df["date"] = trades_df["date"].dt.strftime("%Y-%m-%d")
    metrics = compute_performance_metrics(ledger.frame["net_return"], risk_free=rf)

    split_df = pd.DataFrame(split_records)

    def _summarise_horizons(series: pd.Series, horizons: Iterable[int]) -> pd.DataFrame:
        records: list[dict[str, float | int]] = []
        for horizon in horizons:
            if horizon <= 0 or len(series) < horizon:
                continue
            rolling = (
                (1.0 + series)
                .rolling(window=horizon)
                .apply(lambda arr: float(np.prod(arr) - 1.0), raw=True)
            )
            rolling = rolling.dropna()
            if rolling.empty:
                continue
            std = float(rolling.std(ddof=1))
            sharpe_equiv = (
                float(rolling.mean() / std * np.sqrt(252.0 / horizon))
                if std > 0
                else 0.0
            )
            records.append(
                {
                    "horizon_days": int(horizon),
                    "avg_return": float(rolling.mean()),
                    "median_return": float(rolling.median()),
                    "best_return": float(rolling.max()),
                    "worst_return": float(rolling.min()),
                    "sharpe_equiv": sharpe_equiv,
                }
            )
        return pd.DataFrame(records)

    horizon_df = _summarise_horizons(
        ledger.frame["net_return"], config.evaluation_horizons
    )

    return ledger, weights_df, trades_df, metrics, notes, split_df, horizon_df


def run_backtest(
    config_path: str | Path | None = None,
    *,
    dry_run: bool = True,
    settings: Settings | None = None,
) -> BacktestResult:
    """Execute the backtest described by ``config_path``."""

    settings = settings or get_settings()
    candidate = config_path or (settings.configs_dir / DEFAULT_BACKTEST_CONFIG)
    resolved = resolve_config_path(candidate, settings=settings)
    config = _load_config(resolved, settings)

    result = BacktestResult(
        config_path=resolved,
        environment=settings.environment,
        base_currency=config.base_currency,
        dry_run=dry_run,
    )

    if dry_run:
        return result

    returns, prices, rf = _load_market_data(config)
    (
        ledger,
        weights_df,
        trades_df,
        metrics,
        notes,
        split_df,
        horizon_df,
    ) = _run_simulation(returns, prices, rf, config)

    result.metrics = metrics
    result.ledger = ledger
    result.weights = weights_df
    result.trades = trades_df
    result.split_metrics = split_df
    result.horizon_metrics = horizon_df
    result.notes = notes
    result.dry_run = False

    # Compute walk-forward summary statistics
    if split_df is not None and not split_df.empty:
        try:
            wf_summary = compute_wf_summary_stats(split_df)
            result.walkforward_summary = wf_summary
        except (ValueError, KeyError) as exc:
            result.notes.append(f"Failed to compute walkforward summary: {exc}")

    return result
