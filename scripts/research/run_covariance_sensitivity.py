#!/usr/bin/env python
"""Compare mean-variance strategies under different covariance estimators.

For cada estimador calculamos µ via shrunk mean (strength 0.5) e resolvemos
um programa média-variância com λ=4, max 10% por ativo, custos de 30 bps e
rebalanço 252/21 dias. Os resultados são salvos em
``results/cov_sensitivity/metrics.csv`` e ``returns.csv``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet

from itau_quant.evaluation.oos import StrategySpec, compare_baselines, default_strategies
from itau_quant.estimators.cov import (
    ledoit_wolf_shrinkage,
    nonlinear_shrinkage,
    project_to_psd,
    sample_cov,
    tyler_m_estimator,
)
from itau_quant.estimators.mu import shrunk_mean
from itau_quant.optimization.core.mv_qp import MeanVarianceConfig, solve_mean_variance

OUTPUT_DIR = Path("results") / "cov_sensitivity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_returns() -> pd.DataFrame:
    candidates = [
        Path("data") / "processed" / "returns_arara.parquet",
        Path("results") / "baselines" / "baseline_returns_oos.parquet",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        if "equal_weight" in df.columns:
            continue
        clean = df.astype(float).dropna(how="all")
        mask = clean.count() >= 400  # garantir dados suficientes (aprox 18m)
        filtered = clean.loc[:, mask]
        if filtered.empty:
            continue
        filtered = filtered.sort_index()
        filtered = filtered.ffill()
        filtered = filtered.dropna(how="any")
        return filtered
    raise FileNotFoundError("Nenhum painel de retornos encontrado.")


def covariance_estimator(method: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    method = method.lower()

    def _estimate(returns: pd.DataFrame) -> pd.DataFrame:
        clean = returns.dropna()
        if clean.empty:
            raise ValueError("Sem observações para estimar covariância.")
        cols = clean.columns
        if method == "sample":
            cov = sample_cov(clean)
        elif method == "ledoit_wolf":
            cov, _ = ledoit_wolf_shrinkage(clean)
        elif method == "nonlinear":
            cov = nonlinear_shrinkage(clean)
        elif method == "tyler":
            cov = tyler_m_estimator(clean)
        elif method == "mcd":
            # Minimum Covariance Determinant (scikit-learn) fornece matriz robusta.
            mcd = MinCovDet(assume_centered=False, random_state=1234).fit(clean.values)
            cov = pd.DataFrame(mcd.covariance_, index=cols, columns=cols)
            cov = project_to_psd(cov, epsilon=1e-6)
        elif method == "pca3":
            values = clean.sub(clean.mean()).to_numpy(dtype=float)
            sample = np.cov(values, rowvar=False, ddof=1)
            eigvals, eigvecs = np.linalg.eigh(sample)
            order = np.argsort(eigvals)[::-1]
            top = order[: min(3, len(eigvals))]
            approx = eigvecs[:, top] @ np.diag(eigvals[top]) @ eigvecs[:, top].T
            resid_var = np.diag(sample - approx)
            resid_var = np.clip(resid_var, 0.0, None)
            approx = approx + np.diag(resid_var)
            cov = pd.DataFrame(project_to_psd(approx, epsilon=1e-6), index=cols, columns=cols)
        else:
            raise ValueError(f"covariance method '{method}' não suportado.")
        return cov.reindex(index=cols, columns=cols).astype(float)

    return _estimate


def make_mv_strategy(name: str, cov_fn: Callable[[pd.DataFrame], pd.DataFrame]) -> StrategySpec:
    def builder(train_returns: pd.DataFrame, previous: pd.Series | None) -> pd.Series:
        mu_daily = shrunk_mean(train_returns, strength=0.5, prior=0.0)
        mu = mu_daily * 252.0
        cov_daily = cov_fn(train_returns)
        cov = cov_daily * 252.0

        prev = (
            previous.reindex(train_returns.columns).fillna(0.0)
            if previous is not None
            else pd.Series(0.0, index=train_returns.columns)
        )

        config = MeanVarianceConfig(
            risk_aversion=4.0,
            turnover_penalty=0.0,
            turnover_cap=None,
            lower_bounds=pd.Series(0.0, index=train_returns.columns),
            upper_bounds=pd.Series(0.10, index=train_returns.columns),
            previous_weights=prev,
            cost_vector=None,
            solver="CLARABEL",
        )
        result = solve_mean_variance(mu, cov, config)
        weights = result.weights.reindex(train_returns.columns).clip(lower=0.0, upper=0.10)
        if weights.sum() == 0:
            weights = pd.Series(1.0 / len(weights), index=train_returns.columns)
        else:
            weights = weights / weights.sum()
        return weights

    return StrategySpec(name, builder)


def main() -> None:
    returns = load_returns()
    print(
        f"Dados: {returns.index.min().date()} → {returns.index.max().date()}, "
        f"{returns.shape[1]} ativos"
    )

    base_strategies = default_strategies(max_position=0.10)
    estimators = {
        "mv_sample": covariance_estimator("sample"),
        "mv_ledoit": covariance_estimator("ledoit_wolf"),
        "mv_nonlinear": covariance_estimator("nonlinear"),
        "mv_tyler": covariance_estimator("tyler"),
        "mv_mcd": covariance_estimator("mcd"),
        "mv_pca3": covariance_estimator("pca3"),
    }
    for name, fn in estimators.items():
        base_strategies.append(make_mv_strategy(name, fn))

    result = compare_baselines(
        returns,
        strategies=base_strategies,
        train_window=252,
        test_window=21,
        purge_window=5,
        embargo_window=5,
        costs_bps=30.0,
        max_position=0.10,
        bootstrap_iterations=None,
    )

    metrics = result.metrics.sort_values("sharpe", ascending=False)
    metrics.to_csv(OUTPUT_DIR / "metrics.csv")
    result.returns.to_parquet(OUTPUT_DIR / "returns.parquet")
    print("Métricas salvas em results/cov_sensitivity/metrics.csv")


if __name__ == "__main__":
    main()
