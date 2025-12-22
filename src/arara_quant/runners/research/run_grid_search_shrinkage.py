#!/usr/bin/env python
"""
PRISM-R - Grid Search para Calibração de Shrinkage e Regularization

Testa combinações de:
- μ shrinkage (γ)
- Target volatility
- Turnover cap
- Ridge penalty

Objetivo: encontrar config que bate 1/N + 0.2 Sharpe OOS
"""

import itertools
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from arara_quant.config import get_settings
from arara_quant.data import get_arara_universe

SETTINGS = get_settings()

print("=" * 80)
print("  PRISM-R - Grid Search: Shrinkage + Regularization")
print("  Meta: Sharpe OOS ≥ 1.25 (1/N + 0.20)")
print("=" * 80)
print()

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

TICKERS = get_arara_universe() + ["BITO"]

START_DATE = datetime.now() - timedelta(days=5 * 365)
END_DATE = datetime.now()

TRAIN_WINDOW = 252  # 1 year
TEST_WINDOW = 21  # 1 month
MAX_POSITION = 0.10  # 10% max per asset
TRANSACTION_COST = 0.0030  # 30 bps round-trip

# Grid parameters
GAMMA_VALUES = [0.6, 0.75, 0.9]  # Bayesian shrinkage intensity
VOL_TARGETS = [0.08, 0.10, 0.12]  # Target annual volatility
TURNOVER_CAPS = [0.10, 0.125]  # Monthly turnover caps
RIDGE_VALUES = [0.0, 5e-4]  # Ridge penalty

print("📊 Grid configuration:")
print(f"   • γ (shrinkage):   {GAMMA_VALUES}")
print(f"   • Vol targets:     {[f'{v:.0%}' for v in VOL_TARGETS]}")
print(f"   • TO caps:         {[f'{t:.1%}' for t in TURNOVER_CAPS]}")
print(f"   • Ridge:           {RIDGE_VALUES}")
print(f"   • Total configs:   {len(GAMMA_VALUES) * len(VOL_TARGETS) * len(TURNOVER_CAPS) * len(RIDGE_VALUES)}")
print()

# ============================================================================
# [1] CARREGAR DADOS
# ============================================================================

print("📥 [1/4] Carregando dados históricos...")
print(f"   Período: {START_DATE.date()} a {END_DATE.date()}")

data = yf.download(
    TICKERS,
    start=START_DATE,
    end=END_DATE,
    progress=False,
    auto_adjust=True,
)

if "Close" in data.columns:
    prices = data["Close"]
elif isinstance(data.columns, pd.MultiIndex):
    prices = data.xs("Close", level=0, axis=1)
else:
    prices = data

if isinstance(prices, pd.Series):
    prices = prices.to_frame()

prices = prices.dropna(axis=1, how="all")
valid_tickers = list(prices.columns)

print(f"   ✅ {len(prices)} dias, {len(valid_tickers)} ativos")
print()

# ============================================================================
# [2] CALCULAR RETORNOS
# ============================================================================

print("📊 [2/4] Calculando retornos...")

returns = prices.pct_change().dropna()
returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")

print(f"   ✅ {len(returns)} observações")
print()

# ============================================================================
# [3] DEFINIR FUNÇÃO DE BACKTEST
# ============================================================================

print("⚙️  [3/4] Preparando backtest walk-forward...")

from arara_quant.estimators.cov import ledoit_wolf_shrinkage
from arara_quant.estimators.mu import huber_mean
from arara_quant.estimators.mu_robust import combined_shrinkage
from arara_quant.optimization.core.mv_qp import MeanVarianceConfig, solve_mean_variance


def run_backtest(
    returns_df: pd.DataFrame,
    gamma: float,
    vol_target: float,
    turnover_cap: float,
    ridge: float,
) -> dict:
    """Run walk-forward backtest with given parameters."""

    portfolio_returns = []
    turnovers = []
    prev_weights = None

    n = len(returns_df)
    if n < TRAIN_WINDOW + TEST_WINDOW:
        return {
            "sharpe": np.nan,
            "return_annual": np.nan,
            "vol_annual": np.nan,
            "total_return": np.nan,
            "max_dd": np.nan,
            "avg_turnover": np.nan,
            "n_periods": 0,
        }

    for i in range(TRAIN_WINDOW, n - TEST_WINDOW, TEST_WINDOW):
        train = returns_df.iloc[i - TRAIN_WINDOW : i]
        test = returns_df.iloc[i : i + TEST_WINDOW]

        assets = list(train.columns)

        # Estimate μ with Huber
        try:
            mu_huber = huber_mean(train, delta=1.5) * 252
        except:
            continue

        # Estimate Σ
        try:
            sigma, _ = ledoit_wolf_shrinkage(train)
            sigma_annual = sigma * 252
        except:
            continue

        # Apply shrinkage
        mu_shrunk = combined_shrinkage(
            mu_huber,
            sigma_annual,
            T=len(train),
            prior=0.0,
            gamma=gamma,
            alpha=0.5,  # Fixed blend
        )

        # Portfolio optimization
        if prev_weights is None:
            prev_weights = pd.Series(0.0, index=assets)
        else:
            prev_weights = prev_weights.reindex(assets).fillna(0.0)

        config = MeanVarianceConfig(
            risk_aversion=1.0,  # Will be auto-calibrated
            turnover_penalty=0.05,
            turnover_cap=turnover_cap,
            lower_bounds=pd.Series(0.0, index=assets),
            upper_bounds=pd.Series(MAX_POSITION, index=assets),
            previous_weights=prev_weights,
            cost_vector=pd.Series(TRANSACTION_COST, index=assets),
            solver="CLARABEL",
            ridge_penalty=ridge,
            target_vol=vol_target,  # Auto-calibrate λ
        )

        try:
            result = solve_mean_variance(mu_shrunk, sigma_annual, config)

            if not result.summary.is_optimal():
                continue

            weights = result.weights
            prev_weights = weights

            # Calculate turnover
            turnover = result.turnover
            turnovers.append(turnover)

            # Simulate test period returns
            test_aligned = test.reindex(columns=weights.index).fillna(0.0)
            period_returns = (test_aligned * weights).sum(axis=1)

            # Subtract transaction costs
            cost_drag = turnover * TRANSACTION_COST / len(test)
            period_returns -= cost_drag

            portfolio_returns.extend(period_returns.values)

        except Exception:
            continue

    if len(portfolio_returns) < 10:
        return {
            "sharpe": np.nan,
            "return_annual": np.nan,
            "vol_annual": np.nan,
            "total_return": np.nan,
            "max_dd": np.nan,
            "avg_turnover": np.nan,
            "n_periods": 0,
        }

    port_series = pd.Series(portfolio_returns)

    # Metrics
    ret_annual = port_series.mean() * 252
    vol_annual = port_series.std() * np.sqrt(252)
    sharpe = ret_annual / (vol_annual + 1e-12)

    cumulative = (1 + port_series).cumprod()
    total_return = cumulative.iloc[-1] - 1
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    avg_turnover = np.mean(turnovers) if turnovers else np.nan

    return {
        "sharpe": float(sharpe),
        "return_annual": float(ret_annual),
        "vol_annual": float(vol_annual),
        "total_return": float(total_return),
        "max_dd": float(max_dd),
        "avg_turnover": float(avg_turnover),
        "n_periods": len(portfolio_returns),
    }


# ============================================================================
# [4] RODAR GRID SEARCH
# ============================================================================

print("🔄 [4/4] Rodando grid search...")
print()

configs = list(itertools.product(GAMMA_VALUES, VOL_TARGETS, TURNOVER_CAPS, RIDGE_VALUES))

results = []

for i, (gamma, vol_target, to_cap, ridge) in enumerate(configs, 1):
    print(f"   [{i:2d}/{len(configs)}] γ={gamma:.2f}, vol={vol_target:.0%}, TO={to_cap:.1%}, ridge={ridge:.1e}...", end=" ")

    metrics = run_backtest(returns, gamma, vol_target, to_cap, ridge)

    results.append({
        "gamma": gamma,
        "vol_target": vol_target,
        "turnover_cap": to_cap,
        "ridge": ridge,
        **metrics,
    })

    if np.isnan(metrics["sharpe"]):
        print("❌ Failed")
    else:
        print(f"Sharpe={metrics['sharpe']:.3f}")

print()

# ============================================================================
# [5] ANÁLISE DE RESULTADOS
# ============================================================================

print("=" * 80)
print("  📊 RESULTADOS DO GRID SEARCH")
print("=" * 80)
print()

df_results = pd.DataFrame(results)
df_results = df_results.dropna(subset=["sharpe"])

if len(df_results) == 0:
    print("   ❌ TODAS AS CONFIGURAÇÕES FALHARAM!")
    print("   Verifique os dados e parâmetros.")
    sys.exit(1)

# Sort by Sharpe
df_results = df_results.sort_values("sharpe", ascending=False)

# Top 10
print("🏆 Top 10 configurações (por Sharpe OOS):")
print()

top_cols = ["gamma", "vol_target", "turnover_cap", "ridge", "sharpe", "return_annual", "vol_annual", "avg_turnover"]
top_10 = df_results[top_cols].head(10)

print(top_10.to_string(index=False))
print()

# Best config
best = df_results.iloc[0]

print("✅ Melhor configuração:")
print(f"   • γ (shrinkage):     {best['gamma']:.2f}")
print(f"   • Vol target:        {best['vol_target']:.1%}")
print(f"   • Turnover cap:      {best['turnover_cap']:.1%}")
print(f"   • Ridge:             {best['ridge']:.1e}")
print()
print("   📈 Métricas:")
print(f"   • Sharpe OOS:        {best['sharpe']:.3f}")
print(f"   • Retorno anual:     {best['return_annual']:.2%}")
print(f"   • Vol anual:         {best['vol_annual']:.2%}")
print(f"   • Retorno total:     {best['total_return']:.2%}")
print(f"   • Max DD:            {best['max_dd']:.2%}")
print(f"   • Turnover médio:    {best['avg_turnover']:.2%}")
print()

# Check goal
baseline_sharpe = 1.05  # From previous 1/N result
target_sharpe = baseline_sharpe + 0.20

if best['sharpe'] >= target_sharpe:
    print(f"🎯 META ATINGIDA! Sharpe OOS ({best['sharpe']:.3f}) ≥ alvo ({target_sharpe:.2f})")
else:
    gap = target_sharpe - best['sharpe']
    print(f"⚠️  Meta não atingida. Gap: {gap:.3f} Sharpe points.")
    print(f"   Melhor: {best['sharpe']:.3f} | Alvo: {target_sharpe:.2f}")

print()

# Save results
results_dir = SETTINGS.results_dir
results_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = results_dir / f"grid_search_results_{timestamp}.csv"

df_results.to_csv(output_file, index=False)
print(f"💾 Resultados salvos: {output_file}")

print()
print("=" * 80)
print("  ✅ GRID SEARCH CONCLUÍDO!")
print("=" * 80)
