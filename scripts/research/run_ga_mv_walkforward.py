#!/usr/bin/env python
"""GA-driven hyperparameter search followed by walk-forward validation.

Steps:
1. Load cached ARARA returns (504-day window for in-sample calibration).
2. Estimate μ via Huber and Σ via Ledoit-Wolf.
3. Compute ERC weights to serve as previous portfolio for turnover realism.
4. Run the genetic meta-heuristic to tune (λ, η, τ) and select a subset of assets
   subject to the 20–35 cardinality guardrail.
5. Reuse the best configuration inside a walk-forward evaluation alongside
   equal-weight and risk-parity baselines.
6. Persist calibration outputs and OOS metrics to `results/ga_metaheuristic/`.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from itau_quant.estimators.mu import huber_mean
from itau_quant.estimators.cov import ledoit_wolf_shrinkage
from itau_quant.evaluation.oos import StrategySpec, compare_baselines, default_strategies
from itau_quant.optimization.core.mv_qp import MeanVarianceConfig, solve_mean_variance
from itau_quant.optimization.core.risk_parity import iterative_risk_parity
from itau_quant.optimization.heuristics.metaheuristic import MetaheuristicResult, metaheuristic_outer

RESULTS_DIR = Path("results") / "ga_metaheuristic"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PRINT_WIDTH = 80


def _print_header(title: str) -> None:
    print("=" * PRINT_WIDTH)
    print(f"{title:^{PRINT_WIDTH}}")
    print("=" * PRINT_WIDTH)
    print()


def _annualize(mu_daily: float, var_daily: float) -> tuple[float, float, float]:
    ann_return = (1.0 + mu_daily) ** 252 - 1.0
    ann_vol = np.sqrt(var_daily) * np.sqrt(252.0)
    ann_sharpe = mu_daily / np.sqrt(var_daily) * np.sqrt(252.0) if var_daily > 0 else 0.0
    return float(ann_return), float(ann_vol), float(ann_sharpe)


def run_metaheuristic_calibration(returns: pd.DataFrame) -> tuple[MetaheuristicResult, pd.Series]:
    mu_daily, _ = huber_mean(returns, c=1.5)
    cov_daily, _ = ledoit_wolf_shrinkage(returns)

    assets = mu_daily.index.tolist()
    series = lambda value: pd.Series(value, index=assets, dtype=float)

    cov_annual = cov_daily * 252.0
    erc_weights = iterative_risk_parity(cov_annual).reindex(assets).fillna(0.0)

    base_config = MeanVarianceConfig(
        risk_aversion=6.0,
        turnover_penalty=0.10,
        turnover_cap=0.20,
        lower_bounds=series(0.0),
        upper_bounds=series(0.12),
        previous_weights=erc_weights,
        cost_vector=None,
    )

    ga_config = {
        "seed": 777,
        "generations": 10,
        "population": {
            "size": 28,
            "cardinality": {"min": 20, "max": 35},
            "hyperparams": {
                "lambda": {"choices": [6.0, 9.0, 12.0, 15.0, 18.0]},
                "eta": {"choices": [0.05, 0.10, 0.15, 0.20, 0.25]},
                "tau": {"choices": [0.18, 0.20, 0.25]},
            },
        },
        "mutation": {
            "flip_prob": 0.25,
            "constraints": {"cardinality": {"min": 20, "max": 35}},
        },
        "crossover": {"method": "uniform", "constraints": {"cardinality": {"min": 20, "max": 35}}},
        "selection": {"method": "tournament", "tournament_size": 4},
        "evaluation": {"metric": "sharpe"},
        "elitism": 0.2,
        "constraints": {"cardinality": {"min": 20, "max": 35}},
    }

    result = metaheuristic_outer(
        mu_daily,
        cov_daily,
        base_config,
        ga_config=ga_config,
        turnover_target=(0.05, 0.20),
        cardinality_target=(20, 32),
        penalty_weights={"turnover": 2.0, "cardinality": 5.0},
    )
    return result, erc_weights


def ga_strategy_builder(
    lambda_risk: float,
    eta_turnover: float,
    tau_turnover: float,
    selected_assets: Sequence[str],
    initial_prev: pd.Series,
):
    subset = list(selected_assets)

    def builder(train_returns: pd.DataFrame, prev_weights: pd.Series | None) -> pd.Series:
        data = train_returns.loc[:, subset].dropna(how="all")
        if data.empty:
            return pd.Series(0.0, index=train_returns.columns, dtype=float)

        mu_daily, _ = huber_mean(data, c=1.5)
        cov_daily, _ = ledoit_wolf_shrinkage(data)
        assets = mu_daily.index

        lower = pd.Series(0.0, index=assets, dtype=float)
        upper = pd.Series(0.12, index=assets, dtype=float)
        if prev_weights is None or prev_weights.abs().sum() <= 1e-6:
            previous = initial_prev.reindex(assets).fillna(0.0).astype(float)
        else:
            previous = prev_weights.reindex(assets).fillna(0.0).astype(float)

        config = MeanVarianceConfig(
            risk_aversion=lambda_risk,
            turnover_penalty=eta_turnover,
            turnover_cap=tau_turnover,
            lower_bounds=lower,
            upper_bounds=upper,
            previous_weights=previous,
            cost_vector=None,
        )

        result = solve_mean_variance(mu_daily, cov_daily, config)
        weights = result.weights.clip(lower=0.0)
        total = float(weights.sum())
        if total > 0:
            weights = weights / total

        full = pd.Series(0.0, index=train_returns.columns, dtype=float)
        full.loc[weights.index] = weights
        return full

    return builder


def main() -> None:
    _print_header("GA Hyperparameter Search + Walk-Forward Validation")

    returns_all = pd.read_parquet("data/processed/returns_arara.parquet").dropna(axis=0, how="all")
    returns_all = returns_all.sort_index()

    calibration_window = returns_all.tail(504)
    ga_result, erc_weights = run_metaheuristic_calibration(calibration_window)

    metrics = ga_result.metrics
    params = {k: (float(v) if v is not None else None) for k, v in ga_result.params.items()}
    ann_return, ann_vol, ann_sharpe = _annualize(metrics["expected_return"], metrics["variance"])

    print("📌 Meta-heurística concluída:")
    print(f"   • Status: {ga_result.status}")
    print(f"   • λ*: {params.get('lambda')}  η*: {params.get('eta')}  τ*: {params.get('tau')}")
    print(f"   • Cardinalidade: {metrics.get('cardinality')}")
    print(f"   • Turnover vs ERC: {metrics.get('turnover'):.2%}")
    print(f"   • Sharpe (in-sample): {ann_sharpe:.2f}")
    print(f"   • Retorno/Vol (anual): {ann_return:.2%} / {ann_vol:.2%}")
    print()

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "ga_result.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "timestamp": timestamp,
                "params": params,
                "metrics": {
                    "expected_return_daily": metrics.get("expected_return"),
                    "variance_daily": metrics.get("variance"),
                    "turnover": metrics.get("turnover"),
                    "cardinality": metrics.get("cardinality"),
                    "ann_return": ann_return,
                    "ann_vol": ann_vol,
                    "ann_sharpe": ann_sharpe,
                },
                "selected_assets": list(ga_result.selected_assets),
            },
            fh,
            indent=2,
        )

    ga_strategy = StrategySpec(
        name="mv_ga_tuned",
        builder=ga_strategy_builder(
            lambda_risk=params["lambda"],
            eta_turnover=params["eta"],
            tau_turnover=params["tau"],
            selected_assets=ga_result.selected_assets,
            initial_prev=erc_weights.reindex(returns_all.columns).fillna(0.0),
        ),
    )

    strategies = [
        spec for spec in default_strategies(max_position=0.12) if spec.name in {"equal_weight", "risk_parity"}
    ]
    strategies.append(ga_strategy)

    END_DATE = returns_all.index.max()
    START_DATE = END_DATE - timedelta(days=5 * 365)
    filtered_returns = returns_all.loc[START_DATE:END_DATE]
    min_obs = 252 + 50
    valid_columns = [col for col in filtered_returns.columns if filtered_returns[col].count() >= min_obs]
    filtered_returns = filtered_returns.loc[:, valid_columns].dropna(axis=0, how="any")

    oos = compare_baselines(
        filtered_returns,
        strategies=strategies,
        train_window=252,
        test_window=21,
        purge_window=5,
        embargo_window=5,
        costs_bps=30.0,
        max_position=0.12,
        bootstrap_iterations=0,
    )

    metrics_df = oos.metrics.copy()
    if "strategy" in metrics_df.columns:
        metrics_df = metrics_df.set_index("strategy")
    else:
        metrics_df.index = metrics_df.index.astype(str)
        metrics_df.index.name = "strategy"
    metrics_df = metrics_df.loc[[spec.name for spec in strategies]]
    print("📊 Walk-forward (5 anos, custo 30bps):")
    print(metrics_df[["annualized_return", "volatility", "sharpe", "cvar_95", "max_drawdown", "avg_turnover"]])
    print()

    metrics_path = run_dir / "walkforward_metrics.csv"
    metrics_df.to_csv(metrics_path)
    returns_path = run_dir / "returns_oos.parquet"
    oos.returns.to_parquet(returns_path)

    print("💾 Resultados salvos em:")
    print(f"   • {metrics_path}")
    print(f"   • {returns_path}")
    print()
    print("Done ✅")


if __name__ == "__main__":
    main()
