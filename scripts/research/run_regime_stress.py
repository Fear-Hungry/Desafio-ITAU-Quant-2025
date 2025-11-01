#!/usr/bin/env python
"""Evaluate regime-aware risk aversion on stress periods."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from itau_quant.estimators.cov import ledoit_wolf_shrinkage
from itau_quant.estimators.mu import shrunk_mean
from itau_quant.evaluation.oos import (
    StrategySpec,
    compare_baselines,
    default_strategies,
)
from itau_quant.optimization.core.mv_qp import MeanVarianceConfig, solve_mean_variance
from itau_quant.risk.regime import detect_regime, regime_multiplier

OUTPUT_DIR = Path("results") / "regime_stress"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REGIME_CONFIG = {
    "window_days": 63,
    "vol_thresholds": {"calm": 0.06, "stressed": 0.10},
    "drawdown_crash": -0.08,
    "multipliers": {"calm": 0.75, "neutral": 1.0, "stressed": 2.5, "crash": 4.0},
}

PERIODS = {
    "covid_crash": ("2020-02-01", "2020-12-31"),
    "inflation_2022": ("2022-01-01", "2022-12-31"),
    "banking_2023": ("2023-02-01", "2023-08-31"),
}


def load_returns() -> pd.DataFrame:
    candidates = [
        Path("results") / "baselines" / "baseline_returns_oos.parquet",
        Path("data") / "processed" / "returns_full.parquet",
        Path("data") / "processed" / "returns_arara.parquet",
    ]
    for path in candidates:
        if path.exists():
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
            if "equal_weight" in df.columns:
                # baseline parquet already contains strategy returns; we only need assets
                continue
            return df.astype(float)
    raise FileNotFoundError(
        "Nenhum painel de retornos encontrado. Rode o pipeline ou informe data/processed/returns_arara.parquet."
    )


def make_regime_mv_strategy(
    *,
    base_lambda: float,
    max_position: float,
    shrink_strength: float,
    detection_config: dict[str, object],
) -> StrategySpec:
    """Return a strategy spec that adapts risk aversion to the detected regime."""

    def builder(train_returns: pd.DataFrame, previous: pd.Series | None) -> pd.Series:
        if previous is None:
            clean_prev = pd.Series(0.0, index=train_returns.columns)
        else:
            clean_prev = previous.reindex(train_returns.columns).fillna(0.0)

        snapshot = detect_regime(train_returns, config=detection_config)
        multiplier = regime_multiplier(snapshot, detection_config)
        lambda_adj = base_lambda * multiplier

        mu_daily = shrunk_mean(train_returns, strength=shrink_strength, prior=0.0)
        mu = mu_daily * 252.0
        cov_daily, _ = ledoit_wolf_shrinkage(train_returns)
        cov = cov_daily * 252.0

        bounds = pd.Series(max_position, index=train_returns.columns, dtype=float)
        config = MeanVarianceConfig(
            risk_aversion=lambda_adj,
            turnover_penalty=0.0,
            turnover_cap=None,
            lower_bounds=pd.Series(0.0, index=train_returns.columns, dtype=float),
            upper_bounds=bounds,
            previous_weights=clean_prev,
            cost_vector=None,
            solver="CLARABEL",
        )
        result = solve_mean_variance(mu, cov, config)
        weights = result.weights.reindex(train_returns.columns).fillna(0.0)
        weights = weights.clip(lower=0.0, upper=max_position)
        if weights.sum() == 0:
            weights = pd.Series(1.0 / len(weights), index=train_returns.columns)
        else:
            weights = weights / weights.sum()
        return weights

    return StrategySpec("regime_mv", builder)


def run_period_analysis(name: str, returns: pd.DataFrame) -> None:
    start, end = PERIODS[name]
    period_slice = returns.loc[start:end]
    if period_slice.empty:
        print(f"⚠️  Nenhum dado disponível para {name} ({start} → {end}).")
        return
    valid_mask = period_slice.count() >= 63
    period_slice = period_slice.loc[:, valid_mask]
    period_slice = period_slice.dropna(how="all")
    if period_slice.empty or period_slice.shape[1] < 3:
        print(f"⚠️  Após filtrar ativos com histórico, janela de {name} ficou vazia.")
        return

    strategy_list = default_strategies(max_position=0.10)
    strategy_list.append(
        make_regime_mv_strategy(
            base_lambda=4.0,
            max_position=0.10,
            shrink_strength=0.5,
            detection_config=REGIME_CONFIG,
        )
    )

    oos = compare_baselines(
        period_slice,
        strategies=strategy_list,
        train_window=189,
        test_window=21,
        purge_window=5,
        embargo_window=5,
        costs_bps=30.0,
        max_position=0.10,
        bootstrap_iterations=None,
    )

    if oos.returns.empty:
        print(f"⚠️  Janela {name} não possui splits suficientes após filtros. Pulando.")
        return

    metrics = oos.metrics.sort_values("sharpe", ascending=False)
    output = OUTPUT_DIR / f"{name}_metrics.csv"
    metrics.to_csv(output)
    print(f"📝 {name}: métricas salvas em {output}")

    returns_path = OUTPUT_DIR / f"{name}_returns.parquet"
    oos.returns.to_parquet(returns_path)


def main() -> None:
    returns = load_returns()
    print(f"Dados carregados: {returns.index.min().date()} → {returns.index.max().date()}, {returns.shape[1]} ativos")
    for name in PERIODS:
        run_period_analysis(name, returns)
    print("Done ✅")


if __name__ == "__main__":
    main()
