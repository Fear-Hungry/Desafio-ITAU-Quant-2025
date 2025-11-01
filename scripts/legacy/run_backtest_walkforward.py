#!/usr/bin/env python
"""
PRISM-R - Portfolio Risk Intelligence System
Backtest Walk-Forward com dados reais

Este script roda um backtest completo com validação temporal (walk-forward).
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

print("=" * 80)
print("  PRISM-R - Backtest Walk-Forward")
print("  Validação Out-of-Sample com Dados Reais")
print("=" * 80)
print()

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

# Universo simplificado para backtest mais rápido
TICKERS = [
    # Core portfolio
    "SPY",  # US Large Cap
    "QQQ",  # US Tech
    "EFA",  # Developed Intl
    "VWO",  # Emerging Markets
    "TLT",  # Long-term Treasuries
    "IEF",  # Intermediate Treasuries
    "LQD",  # Investment Grade Corp
    "GLD",  # Gold
    "DBC",  # Commodities
    "VNQ",  # Real Estate
]

# Período de backtest
END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=365 * 5)  # 5 anos

# Walk-forward parameters
TRAIN_WINDOW = 252  # 1 ano de treino
TEST_WINDOW = 21  # 1 mês de teste
REBALANCE_FREQ = 21  # rebalancear mensalmente

# Optimizer config
RISK_AVERSION = 3.0
MAX_POSITION = 0.30

print("📊 Configuração Backtest:")
print(f"   • Universo: {len(TICKERS)} ativos")
print(f"   • Período: {START_DATE.date()} a {END_DATE.date()}")
print(f"   • Train window: {TRAIN_WINDOW} dias")
print(f"   • Test window: {TEST_WINDOW} dias")
print(f"   • Rebalance: a cada {REBALANCE_FREQ} dias")
print()

# ============================================================================
# 1. CARREGAR DADOS
# ============================================================================
print("📥 [1/4] Carregando dados históricos...")

try:
    import yfinance as yf

    data = yf.download(
        tickers=TICKERS,
        start=START_DATE - timedelta(days=400),  # buffer
        end=END_DATE,
        progress=False,
        auto_adjust=True,
    )

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data

    prices = prices.dropna(how="all")
    prices = prices.ffill().bfill()

    # Filtrar ativos válidos
    valid_tickers = []
    for ticker in TICKERS:
        if (
            ticker in prices.columns
            and prices[ticker].notna().sum() >= TRAIN_WINDOW + 50
        ):
            valid_tickers.append(ticker)

    prices = prices[valid_tickers]
    returns = prices.pct_change().dropna()

    print(f"   ✅ Dados carregados: {len(returns)} dias, {len(valid_tickers)} ativos")
    print(f"   ✅ Período: {returns.index[0].date()} a {returns.index[-1].date()}")
    print()

except Exception as e:
    print(f"   ❌ Erro: {e}")
    sys.exit(1)

# ============================================================================
# 2. CRIAR WALK-FORWARD SPLITS
# ============================================================================
print("🔀 [2/4] Criando splits walk-forward...")

from itau_quant.backtesting.walk_forward import generate_walk_forward_splits

try:
    splits = list(
        generate_walk_forward_splits(
            returns.index,
            train_window=TRAIN_WINDOW,
            test_window=TEST_WINDOW,
            purge_window=2,
            embargo_window=0,
        )
    )

    print(f"   ✅ {len(splits)} períodos de teste criados")
    print(f"   ✅ Primeiro teste: {splits[0].test_index[0].date()}")
    print(f"   ✅ Último teste:   {splits[-1].test_index[-1].date()}")
    print()

except Exception as e:
    print(f"   ❌ Erro: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 3. RODAR BACKTEST
# ============================================================================
print("🔄 [3/4] Rodando backtest walk-forward...")

from itau_quant.estimators.cov import ledoit_wolf_shrinkage
from itau_quant.estimators.mu import mean_return
from itau_quant.optimization.core.mv_qp import MeanVarianceConfig, solve_mean_variance

# Armazenar resultados
portfolio_returns = []
portfolio_weights_history = []
dates = []

initial_capital = 1.0
nav = initial_capital
nav_series = []

print(f"   Processando {len(splits)} períodos...")

for i, split in enumerate(splits):
    # Train period
    train_returns = returns.loc[split.train_index]
    test_returns = returns.loc[split.test_index]

    if len(train_returns) < TRAIN_WINDOW // 2:
        continue

    try:
        # Estimar parâmetros no train set
        mu = mean_return(train_returns) * 252
        sigma, _ = ledoit_wolf_shrinkage(train_returns)
        sigma = sigma * 252

        # Otimizar
        config = MeanVarianceConfig(
            risk_aversion=RISK_AVERSION,
            turnover_penalty=0.05,
            turnover_cap=None,
            lower_bounds=pd.Series(0.0, index=valid_tickers),
            upper_bounds=pd.Series(MAX_POSITION, index=valid_tickers),
            previous_weights=pd.Series(0.0, index=valid_tickers),
            cost_vector=None,
            solver="CLARABEL",
        )

        result = solve_mean_variance(mu, sigma, config)

        if not result.summary.is_optimal():
            print(f"      Warning: período {i + 1} não optimal")
            continue

        weights = result.weights

        # Aplicar pesos no test set
        test_portfolio_returns = (test_returns * weights).sum(axis=1)

        # Atualizar NAV
        for ret in test_portfolio_returns:
            nav *= 1 + ret
            nav_series.append(nav)

        portfolio_returns.extend(test_portfolio_returns.tolist())
        dates.extend(test_returns.index.tolist())
        portfolio_weights_history.append(
            {
                "date": split.test_index[0],
                "weights": weights.to_dict(),
            }
        )

        if (i + 1) % 10 == 0:
            print(f"      Processado {i + 1}/{len(splits)} períodos... NAV={nav:.2f}")

    except Exception as e:
        print(f"      Erro no período {i + 1}: {e}")
        continue

print("   ✅ Backtest concluído!")
print(f"   ✅ {len(portfolio_returns)} retornos diários calculados")
print()

# ============================================================================
# 4. CALCULAR MÉTRICAS
# ============================================================================
print("📊 [4/4] Calculando métricas de performance...")

portfolio_returns_series = pd.Series(portfolio_returns, index=dates)

# Retornos cumulativos
cumulative_returns = (1 + portfolio_returns_series).cumprod()
total_return = cumulative_returns.iloc[-1] - 1

# Retorno anualizado
n_years = len(portfolio_returns_series) / 252
annualized_return = (1 + total_return) ** (1 / n_years) - 1

# Volatilidade anualizada
annualized_vol = portfolio_returns_series.std() * np.sqrt(252)

# Sharpe ratio
sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0

# Max drawdown
running_max = cumulative_returns.expanding().max()
drawdown = (cumulative_returns - running_max) / running_max
max_drawdown = drawdown.min()

# Sortino ratio
downside_returns = portfolio_returns_series[portfolio_returns_series < 0]
downside_vol = (
    downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0001
)
sortino = annualized_return / downside_vol if downside_vol > 0 else 0

# Win rate
win_rate = (portfolio_returns_series > 0).sum() / len(portfolio_returns_series)

print("   📈 Métricas Out-of-Sample:")
print(f"      • Período:               {dates[0].date()} a {dates[-1].date()}")
print(f"      • Dias de trading:       {len(portfolio_returns_series)}")
print(f"      • Retorno total:         {total_return:+.2%}")
print(f"      • Retorno anualizado:    {annualized_return:+.2%}")
print(f"      • Volatilidade anual:    {annualized_vol:.2%}")
print(f"      • Sharpe Ratio:          {sharpe:.2f}")
print(f"      • Sortino Ratio:         {sortino:.2f}")
print(f"      • Max Drawdown:          {max_drawdown:.2%}")
print(f"      • Win Rate:              {win_rate:.2%}")
print(f"      • NAV Final:             {nav:.2f}")
print()

# Comparar com buy-and-hold SPY
if "SPY" in returns.columns:
    spy_returns = returns.loc[dates, "SPY"]
    spy_cumulative = (1 + spy_returns).cumprod()
    spy_total_return = spy_cumulative.iloc[-1] - 1
    spy_annual_return = (1 + spy_total_return) ** (1 / n_years) - 1
    spy_annual_vol = spy_returns.std() * np.sqrt(252)
    spy_sharpe = spy_annual_return / spy_annual_vol if spy_annual_vol > 0 else 0

    print("   📊 Comparação com SPY (Buy & Hold):")
    print(f"      • SPY Retorno total:     {spy_total_return:+.2%}")
    print(f"      • SPY Retorno anual:     {spy_annual_return:+.2%}")
    print(f"      • SPY Volatilidade:      {spy_annual_vol:.2%}")
    print(f"      • SPY Sharpe:            {spy_sharpe:.2f}")
    print(
        f"      • Alpha vs SPY:          {annualized_return - spy_annual_return:+.2%}"
    )
    print(f"      • Sharpe Improvement:    {sharpe - spy_sharpe:+.2f}")
    print()

# ============================================================================
# SALVAR RESULTADOS
# ============================================================================
print("💾 Salvando resultados...")

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Salvar série de retornos
returns_df = pd.DataFrame(
    {
        "date": portfolio_returns_series.index,
        "return": portfolio_returns_series.values,
    }
)
returns_file = output_dir / f"backtest_returns_{timestamp}.csv"
returns_df.to_csv(returns_file, index=False)
print(f"   ✅ Retornos salvos: {returns_file}")

# Salvar métricas
metrics = {
    "timestamp": timestamp,
    "n_assets": len(valid_tickers),
    "n_periods": len(splits),
    "total_return": total_return,
    "annualized_return": annualized_return,
    "volatility": annualized_vol,
    "sharpe_ratio": sharpe,
    "sortino_ratio": sortino,
    "max_drawdown": max_drawdown,
    "win_rate": win_rate,
    "final_nav": nav,
}

metrics_df = pd.DataFrame([metrics])
metrics_file = output_dir / f"backtest_metrics_{timestamp}.csv"
metrics_df.to_csv(metrics_file, index=False)
print(f"   ✅ Métricas salvas: {metrics_file}")

print()
print("=" * 80)
print("  ✅ BACKTEST WALK-FORWARD CONCLUÍDO!")
print("=" * 80)
print()
print("🎯 Resultado final:")
print(f"   • Retorno anualizado: {annualized_return:+.2%}")
print(f"   • Sharpe Ratio: {sharpe:.2f}")
print(f"   • Max Drawdown: {max_drawdown:.2%}")
print()
print("📁 Arquivos gerados:")
print(f"   • {returns_file}")
print(f"   • {metrics_file}")
print()
