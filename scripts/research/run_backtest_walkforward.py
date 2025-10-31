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
from itau_quant.data import get_arara_universe

print("=" * 80)
print("  PRISM-R - Backtest Walk-Forward")
print("  Validação Out-of-Sample com Dados Reais")
print("=" * 80)
print()

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

# Universo completo
TICKERS = get_arara_universe()

# Período de backtest
END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=365 * 5)  # 5 anos

# Walk-forward parameters
TRAIN_WINDOW = 252  # 1 ano de treino
TEST_WINDOW = 21  # 1 mês de teste
REBALANCE_FREQ = 21  # rebalancear mensalmente

# Optimizer config
RISK_AVERSION = 8.0
MAX_POSITION = 0.20

print(f"📊 Configuração Backtest:")
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

    spy_ma200 = None
    spy_ma50 = None
    spy_ret126 = None
    spy_ret63 = None
    if "SPY" in prices.columns:
        spy_ma200 = prices["SPY"].rolling(200).mean()
        spy_ma50 = prices["SPY"].rolling(50).mean()
        spy_cum126 = prices["SPY"] / prices["SPY"].shift(126) - 1
        spy_cum63 = prices["SPY"] / prices["SPY"].shift(63) - 1
        spy_ret126 = spy_cum126
        spy_ret63 = spy_cum63

    # Filtrar ativos válidos
    valid_tickers = []
    for ticker in TICKERS:
        if (
            ticker in prices.columns
            and prices[ticker].notna().sum() >= TRAIN_WINDOW + 50
        ):
            valid_tickers.append(ticker)

    prices = prices[valid_tickers]
    prices["CASH"] = 1.0
    returns = prices.pct_change().dropna()
    returns["CASH"] = 0.0
    if "CASH" not in valid_tickers:
        valid_tickers.append("CASH")

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

from itau_quant.estimators.mu import mean_return
from itau_quant.estimators.cov import ledoit_wolf_shrinkage
from itau_quant.optimization.core.mv_qp import solve_mean_variance, MeanVarianceConfig

DEFENSIVE_BLEND = 1.0
DEFENSIVE_TEMPLATE = {
    "CASH": 0.40,
    "IEF": 0.30,
    "TLT": 0.15,
    "LQD": 0.10,
    "GLD": 0.05,
}
DRAWDOWN_LIMIT = 0.15
RECOVERY_THRESHOLD = 0.03


# Armazenar resultados
portfolio_returns = []
portfolio_weights_history = []
dates = []

initial_capital = 1.0
nav = initial_capital
nav_series = []
nav_peak = nav
in_drawdown_mode = False

print(f"   Processando {len(splits)} períodos...")

for i, split in enumerate(splits):
    # Train period
    train_returns = returns.loc[split.train_index]
    test_returns = returns.loc[split.test_index]

    if len(train_returns) < TRAIN_WINDOW // 2:
        continue

    try:
        if in_drawdown_mode:
            weights = (
                pd.Series(DEFENSIVE_TEMPLATE)
                .reindex(valid_tickers, fill_value=0.0)
            )
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = pd.Series(1.0 / len(valid_tickers), index=valid_tickers)
        else:
            # Estimar parâmetros no train set
            mu = mean_return(train_returns) * 252
            sigma, _ = ledoit_wolf_shrinkage(train_returns)
            sigma = sigma * 252
            if isinstance(sigma, pd.DataFrame):
                if "CASH" in sigma.index:
                    sigma.loc["CASH", :] = 0.0
                    sigma.loc[:, "CASH"] = 0.0
                    sigma.loc["CASH", "CASH"] = 1e-6
            else:
                sigma = pd.DataFrame(sigma, index=train_returns.columns, columns=train_returns.columns)
                sigma.loc["CASH", :] = 0.0
                sigma.loc[:, "CASH"] = 0.0
                sigma.loc["CASH", "CASH"] = 1e-6

            lower_bounds = pd.Series(0.0, index=valid_tickers)
            upper_bounds = pd.Series(MAX_POSITION, index=valid_tickers)
            if "CASH" in upper_bounds:
                upper_bounds["CASH"] = 1.0
            if "CASH" in lower_bounds:
                lower_bounds["CASH"] = 0.50

            config = MeanVarianceConfig(
                risk_aversion=RISK_AVERSION,
                turnover_penalty=0.05,
                turnover_cap=None,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                previous_weights=pd.Series(0.0, index=valid_tickers),
                cost_vector=None,
                solver="CLARABEL",
            )

            result = solve_mean_variance(mu, sigma, config)

            if not result.summary.is_optimal():
                print(f"      Warning: período {i + 1} não optimal")
                continue

            weights = result.weights

            risk_off = False
            if "SPY" in prices.columns:
                test_start = split.test_index[0]
                if test_start in prices.index:
                    spy_price = prices.at[test_start, "SPY"]
                    spy_ma = (
                        spy_ma200.at[test_start]
                        if spy_ma200 is not None
                        else np.nan
                    )
                    spy_ma_fast = (
                        spy_ma50.at[test_start]
                        if spy_ma50 is not None
                        else np.nan
                    )
                    spy_ret = (
                        spy_ret126.at[test_start]
                        if spy_ret126 is not None
                        else np.nan
                    )
                    spy_ret_short = (
                        spy_ret63.at[test_start]
                        if spy_ret63 is not None
                        else np.nan
                    )
                    if pd.notna(spy_ma) and spy_price < spy_ma:
                        risk_off = True
                    if pd.notna(spy_ma_fast) and spy_price < spy_ma_fast:
                        risk_off = True
                    if pd.notna(spy_ret) and spy_ret < 0:
                        risk_off = True
                    if pd.notna(spy_ret) and spy_ret < -0.05:
                        risk_off = True
                    if pd.notna(spy_ret_short) and spy_ret_short < -0.02:
                        risk_off = True

            if risk_off:
                defensive_weights = (
                    pd.Series(DEFENSIVE_TEMPLATE)
                    .reindex(valid_tickers, fill_value=0.0)
                )
                if defensive_weights.sum() > 0:
                    defensive_weights = defensive_weights / defensive_weights.sum()
                    weights = (
                        (1 - DEFENSIVE_BLEND) * weights
                        + DEFENSIVE_BLEND * defensive_weights
                    )
                    weights = weights.clip(lower=0)
                    if weights.sum() > 0:
                        weights = weights / weights.sum()

        # Aplicar pesos no test set
        test_portfolio_returns = (test_returns * weights).sum(axis=1)

        # Atualizar NAV
        for ret in test_portfolio_returns:
            nav *= 1 + ret
            nav_series.append(nav)
            nav_peak = max(nav_peak, nav)
            current_drawdown = nav / nav_peak - 1
            if current_drawdown < -DRAWDOWN_LIMIT:
                in_drawdown_mode = True
            elif in_drawdown_mode and current_drawdown > -RECOVERY_THRESHOLD:
                in_drawdown_mode = False

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

print(f"   ✅ Backtest concluído!")
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

print(f"   📈 Métricas Out-of-Sample:")
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

    print(f"   📊 Comparação com SPY (Buy & Hold):")
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
print(f"🎯 Resultado final:")
print(f"   • Retorno anualizado: {annualized_return:+.2%}")
print(f"   • Sharpe Ratio: {sharpe:.2f}")
print(f"   • Max Drawdown: {max_drawdown:.2%}")
print()
print(f"📁 Arquivos gerados:")
print(f"   • {returns_file}")
print(f"   • {metrics_file}")
print()
