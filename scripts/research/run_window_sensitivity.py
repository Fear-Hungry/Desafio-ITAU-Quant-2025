#!/usr/bin/env python
"""Compare baseline strategies under different estimation windows.

The script reuses the walk-forward evaluator while varying the training window
length (126, 252, 504 dias) to show how sensitive the strategies are to the
amount of historical data used for μ e Σ. Outputs land in
``results/window_sensitivity/``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from itau_quant.evaluation.oos import compare_baselines, default_strategies

OUTPUT_DIR = Path("results") / "window_sensitivity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_WINDOWS = [126, 252, 504]


def load_returns() -> pd.DataFrame:
    candidates = [
        Path("data") / "processed" / "returns_arara.parquet",
        Path("results") / "baselines" / "baseline_returns_oos.parquet",
    ]
    min_count = max(TRAIN_WINDOWS) + 42  # train + safety margin
    for path in candidates:
        if path.exists():
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
            if "equal_weight" in df.columns:
                # Baseline parquet is already aggregated strategy returns; skip.
                continue
            clean = df.astype(float).dropna(how="all")
            mask = clean.count() >= min_count
            filtered = clean.loc[:, mask]
            if filtered.empty:
                raise ValueError("Painel sem ativos com histórico suficiente.")
            return filtered
    raise FileNotFoundError("Nenhum painel de retornos encontrado.")


def run_window(window: int, returns: pd.DataFrame) -> None:
    strategies = default_strategies(max_position=0.10)
    result = compare_baselines(
        returns,
        strategies=strategies,
        train_window=window,
        test_window=21,
        purge_window=5,
        embargo_window=5,
        costs_bps=30.0,
        max_position=0.10,
        bootstrap_iterations=None,
    )

    metrics = result.metrics.sort_values("sharpe", ascending=False)
    metrics.to_csv(OUTPUT_DIR / f"metrics_window_{window}.csv")
    result.returns.to_parquet(OUTPUT_DIR / f"returns_window_{window}.parquet")


def main() -> None:
    returns = load_returns()
    print(
        f"Dados: {returns.index.min().date()} → {returns.index.max().date()}, "
        f"{returns.shape[1]} ativos"
    )

    for window in TRAIN_WINDOWS:
        if window + 21 >= len(returns):
            print(f"⚠️  Window {window} muito grande para o painel. Pulando.")
            continue
        print(f"⏱  Rodando walk-forward com janela de treino = {window} dias…")
        run_window(window, returns)
        print(f"   ↳ artefatos em results/window_sensitivity/metrics_window_{window}.csv")

    print("Done ✅")


if __name__ == "__main__":
    main()
