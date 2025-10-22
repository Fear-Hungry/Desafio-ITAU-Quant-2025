#!/usr/bin/env python
"""
Grid search para encontrar o nível ótimo de shrinkage Bayesiano

Testa múltiplos valores de strength (0.0 a 0.7) e identifica o melhor Sharpe OOS
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Carregar dados
from itau_quant.data.sources.yf import download_prices
from itau_quant.data.processing.returns import calculate_returns
from itau_quant.estimators.mu import bayesian_shrinkage_mean
from itau_quant.estimators.cov import ledoit_wolf_shrinkage
from itau_quant.optimization.core.mv_qp import solve_mean_variance, MeanVarianceConfig
from itau_quant.backtesting.walk_forward import generate_walk_forward_splits
from itau_quant.backtesting.metrics import sharpe_ratio

print("=" * 80)
print("  GRID SEARCH: Bayesian Shrinkage Strength")
print("  Objetivo: Encontrar strength ótimo para maximizar Sharpe OOS")
print("=" * 80)
print()

# Configuração
TICKERS = [
    "SPY", "QQQ", "IWM", "VTV", "VUG",  # US Equity
    "EFA", "VGK", "EWJ", "EWU",         # Intl Developed
    "EEM", "VWO", "INDA", "FXI",        # EM
    "IEF", "TLT", "SHY", "LQD", "HYG", "EMB",  # Fixed Income
    "GLD", "SLV", "DBC", "USO",         # Commodities
    "IBIT", "ETHA",                      # Crypto spot
    "VNQ",                               # REIT
]

START_DATE = "2020-10-01"
END_DATE = "2025-10-31"
MAX_POSITION = 0.10
TRANSACTION_COST_BPS = 30
TRAIN_WINDOW = 252
TEST_WINDOW = 21
PURGE_WINDOW = 5
EMBARGO_WINDOW = 5

# Níveis de shrinkage a testar
STRENGTH_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

print("📥 Carregando dados...")
prices = download_prices(TICKERS, start=START_DATE, end=END_DATE)
returns = calculate_log_returns(prices).dropna()
print(f"   ✅ {len(returns)} dias, {len(returns.columns)} ativos")
print()

# Gerar splits
splits = list(
    generate_walk_forward_splits(
        returns.index,
        train_window=TRAIN_WINDOW,
        test_window=TEST_WINDOW,
        purge_window=PURGE_WINDOW,
        embargo_window=EMBARGO_WINDOW,
    )
)
print(f"📊 {len(splits)} períodos walk-forward gerados")
print()

def run_backtest_with_strength(strength):
    """Backtest com nível específico de shrinkage"""
    portfolio_returns = []

    for split in splits:
        train_returns = returns.loc[split.train_index]
        test_returns = returns.loc[split.test_index]

        if train_returns.empty or test_returns.empty:
            continue

        # Estimar μ com shrinkage
        mu_daily = bayesian_shrinkage_mean(train_returns, prior=0.0, strength=strength)
        mu = mu_daily * 252

        # Estimar Σ com Ledoit-Wolf
        cov, _ = ledoit_wolf_shrinkage(train_returns)
        cov_annual = cov * 252

        # Otimizar
        cost_vector = pd.Series(TRANSACTION_COST_BPS / 10000, index=train_returns.columns)

        config = MeanVarianceConfig(
            risk_aversion=4.0,
            turnover_penalty=0.0015,
            turnover_cap=None,
            lower_bounds=pd.Series(0.0, index=train_returns.columns),
            upper_bounds=pd.Series(MAX_POSITION, index=train_returns.columns),
            previous_weights=pd.Series(0.0, index=train_returns.columns),
            cost_vector=cost_vector,
            solver="CLARABEL",
        )

        try:
            result = solve_mean_variance(mu, cov_annual, config)
            if result.summary.is_optimal():
                weights = result.weights
            else:
                weights = pd.Series(1.0 / len(train_returns.columns), index=train_returns.columns)
        except:
            weights = pd.Series(1.0 / len(train_returns.columns), index=train_returns.columns)

        # Calcular retornos OOS
        for date in split.test_index:
            ret = (weights * returns.loc[date]).sum()
            portfolio_returns.append(ret)

    returns_series = pd.Series(portfolio_returns)
    sharpe = sharpe_ratio(returns_series) if len(returns_series) > 0 else 0.0

    return sharpe, returns_series

print("🔍 Testando diferentes níveis de shrinkage...")
print()

results = []
for strength in STRENGTH_LEVELS:
    print(f"   Testando strength={strength:.1f}...", end=" ")
    sharpe, _ = run_backtest_with_strength(strength)
    results.append({"strength": strength, "sharpe_oos": sharpe})
    print(f"Sharpe OOS = {sharpe:.3f}")

print()
print("=" * 80)
print("  RESULTADOS")
print("=" * 80)
print()

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
print()

best = results_df.loc[results_df['sharpe_oos'].idxmax()]
print(f"🏆 Melhor configuração:")
print(f"   • Strength: {best['strength']:.1f}")
print(f"   • Sharpe OOS: {best['sharpe_oos']:.3f}")
print()

# Salvar
output_path = Path("results") / f"shrinkage_grid_search_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
results_df.to_csv(output_path, index=False)
print(f"💾 Resultados salvos: {output_path}")
