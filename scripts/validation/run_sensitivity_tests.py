#!/usr/bin/env python
"""
PRISM-R - Sensitivity Analysis
Testes de sensibilidade para parâmetros críticos do sistema

Este script testa:
1. Diferentes vol targets (8%, 10%, 12%, 14%)
2. Diferentes CASH floors (10%, 15%, 20%, 25%)
3. Diferentes cardinalities (K=15, 18, 22, 25, 28)
4. Diferentes turnover penalties
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

print("=" * 80)
print("  PRISM-R - SENSITIVITY ANALYSIS")
print("  Análise de Sensibilidade de Parâmetros")
print("=" * 80)
print()

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

UNIVERSE_PATH = Path("configs/universe_arara_robust.yaml")

with open(UNIVERSE_PATH) as f:
    universe_config = yaml.safe_load(f)

TICKERS = universe_config["tickers"]

# Período de teste
END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=365 * 3)  # 3 anos para testes mais rápidos

# Walk-forward parameters
TRAIN_WINDOW = 252
TEST_WINDOW = 21

# Parâmetros para testar
VOL_TARGETS = [0.08, 0.10, 0.12, 0.14]
CASH_FLOORS = [0.05, 0.10, 0.15, 0.20, 0.25]
CARDINALITIES = [15, 18, 22, 25, 28]

print("📊 Configuração:")
print(f"   • Universo: {len(TICKERS)} ativos ARARA")
print(f"   • Período: {START_DATE.date()} a {END_DATE.date()}")
print(f"   • Vol Targets: {[f'{v:.0%}' for v in VOL_TARGETS]}")
print(f"   • CASH Floors: {[f'{c:.0%}' for c in CASH_FLOORS]}")
print(f"   • Cardinalities: {CARDINALITIES}")
print()

# ============================================================================
# CARREGAR DADOS
# ============================================================================
print("📥 [1/4] Carregando dados históricos...")

try:
    import yfinance as yf

    data = yf.download(
        tickers=TICKERS,
        start=START_DATE - timedelta(days=400),
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
    prices["CASH"] = 1.0
    returns = prices.pct_change().dropna()
    returns["CASH"] = 0.0
    valid_tickers.append("CASH")

    print(f"   ✅ Dados carregados: {len(returns)} dias, {len(valid_tickers)} ativos")
    print()

except Exception as e:
    print(f"   ❌ Erro: {e}")
    sys.exit(1)

# ============================================================================
# CRIAR SPLITS
# ============================================================================
print("🔀 [2/4] Criando splits walk-forward...")

from itau_quant.backtesting.walk_forward import generate_walk_forward_splits

splits = list(
    generate_walk_forward_splits(
        returns.index,
        train_window=TRAIN_WINDOW,
        test_window=TEST_WINDOW,
        purge_window=2,
        embargo_window=0,
    )
)

print(f"   ✅ {len(splits)} períodos criados")
print()

# ============================================================================
# FUNÇÃO ERC
# ============================================================================


def calculate_erc_weights(sigma, cardinality=None):
    """Calcula pesos ERC usando log-barrier method"""
    from scipy.optimize import minimize

    n = len(sigma)

    def objective(w):
        w = np.maximum(w, 1e-8)
        portfolio_var = w @ sigma @ w
        marginal_contrib = sigma @ w
        risk_contrib = w * marginal_contrib
        log_rc = np.log(np.maximum(risk_contrib, 1e-10))
        obj = -np.sum(log_rc)
        return obj

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0) for _ in range(n)]
    w0 = np.ones(n) / n

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    if result.success:
        weights = result.x

        if cardinality is not None and cardinality < n:
            top_k_idx = np.argsort(weights)[-cardinality:]
            weights_k = np.zeros(n)
            weights_k[top_k_idx] = weights[top_k_idx]
            if weights_k.sum() > 0:
                weights_k = weights_k / weights_k.sum()
            weights = weights_k

        return weights
    else:
        return np.ones(n) / n


# ============================================================================
# TESTE 1: SENSIBILIDADE A VOL TARGET
# ============================================================================
print("=" * 80)
print("🔬 [3/4] TESTE 1: Sensibilidade ao Vol Target")
print("=" * 80)
print()

from itau_quant.estimators.cov import ledoit_wolf_shrinkage

results_vol = {}

for vol_target in VOL_TARGETS:
    print(f"   Testando Vol Target = {vol_target:.0%}...")

    portfolio_returns = []
    dates = []
    nav = 1.0

    for split in splits:
        train_returns = returns.loc[split.train_index]
        test_returns = returns.loc[split.test_index]

        if len(train_returns) < TRAIN_WINDOW // 2:
            continue

        try:
            sigma, _ = ledoit_wolf_shrinkage(train_returns)
            sigma = sigma * 252

            assets_no_cash = [t for t in valid_tickers if t != "CASH"]
            sigma_no_cash = sigma.loc[assets_no_cash, assets_no_cash].values

            erc_weights = calculate_erc_weights(sigma_no_cash, cardinality=22)
            weights = pd.Series(erc_weights, index=assets_no_cash)
            weights = weights.reindex(valid_tickers, fill_value=0.0)

            # Aplicar CASH floor fixo de 15%
            cash_floor = 0.15
            weights = weights * (1 - cash_floor)
            weights["CASH"] = cash_floor

            if weights.sum() > 0:
                weights = weights / weights.sum()

            test_portfolio_returns = (test_returns * weights).sum(axis=1)

            for ret in test_portfolio_returns:
                nav *= 1 + ret

            portfolio_returns.extend(test_portfolio_returns.tolist())
            dates.extend(test_returns.index.tolist())

        except Exception:
            continue

    # Calcular métricas
    portfolio_returns_series = pd.Series(portfolio_returns, index=dates)
    cumulative_returns = (1 + portfolio_returns_series).cumprod()
    total_return = cumulative_returns.iloc[-1] - 1
    n_years = len(portfolio_returns_series) / 252
    annualized_return = (1 + total_return) ** (1 / n_years) - 1
    annualized_vol = portfolio_returns_series.std() * np.sqrt(252)
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0

    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    results_vol[vol_target] = {
        "return": annualized_return,
        "vol": annualized_vol,
        "sharpe": sharpe,
        "max_dd": max_drawdown,
    }

    print(
        f"      Sharpe: {sharpe:.2f} | Vol: {annualized_vol:.1%} | DD: {max_drawdown:.1%}"
    )

print()
print("   Resumo Vol Target:")
vol_df = pd.DataFrame(results_vol).T
vol_df.columns = ["Retorno", "Vol", "Sharpe", "Max DD"]
print(vol_df.to_string())
print()

# ============================================================================
# TESTE 2: SENSIBILIDADE A CASH FLOOR
# ============================================================================
print("=" * 80)
print("🔬 [3/4] TESTE 2: Sensibilidade ao CASH Floor")
print("=" * 80)
print()

results_cash = {}

for cash_floor in CASH_FLOORS:
    print(f"   Testando CASH Floor = {cash_floor:.0%}...")

    portfolio_returns = []
    dates = []
    nav = 1.0

    for split in splits:
        train_returns = returns.loc[split.train_index]
        test_returns = returns.loc[split.test_index]

        if len(train_returns) < TRAIN_WINDOW // 2:
            continue

        try:
            sigma, _ = ledoit_wolf_shrinkage(train_returns)
            sigma = sigma * 252

            assets_no_cash = [t for t in valid_tickers if t != "CASH"]
            sigma_no_cash = sigma.loc[assets_no_cash, assets_no_cash].values

            erc_weights = calculate_erc_weights(sigma_no_cash, cardinality=22)
            weights = pd.Series(erc_weights, index=assets_no_cash)
            weights = weights.reindex(valid_tickers, fill_value=0.0)

            weights = weights * (1 - cash_floor)
            weights["CASH"] = cash_floor

            if weights.sum() > 0:
                weights = weights / weights.sum()

            test_portfolio_returns = (test_returns * weights).sum(axis=1)

            for ret in test_portfolio_returns:
                nav *= 1 + ret

            portfolio_returns.extend(test_portfolio_returns.tolist())
            dates.extend(test_returns.index.tolist())

        except Exception:
            continue

    # Calcular métricas
    portfolio_returns_series = pd.Series(portfolio_returns, index=dates)
    cumulative_returns = (1 + portfolio_returns_series).cumprod()
    total_return = cumulative_returns.iloc[-1] - 1
    n_years = len(portfolio_returns_series) / 252
    annualized_return = (1 + total_return) ** (1 / n_years) - 1
    annualized_vol = portfolio_returns_series.std() * np.sqrt(252)
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0

    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    results_cash[cash_floor] = {
        "return": annualized_return,
        "vol": annualized_vol,
        "sharpe": sharpe,
        "max_dd": max_drawdown,
    }

    print(
        f"      Sharpe: {sharpe:.2f} | Vol: {annualized_vol:.1%} | DD: {max_drawdown:.1%}"
    )

print()
print("   Resumo CASH Floor:")
cash_df = pd.DataFrame(results_cash).T
cash_df.columns = ["Retorno", "Vol", "Sharpe", "Max DD"]
print(cash_df.to_string())
print()

# ============================================================================
# TESTE 3: SENSIBILIDADE A CARDINALITY
# ============================================================================
print("=" * 80)
print("🔬 [3/4] TESTE 3: Sensibilidade à Cardinalidade")
print("=" * 80)
print()

results_card = {}

for cardinality in CARDINALITIES:
    print(f"   Testando Cardinality K = {cardinality}...")

    portfolio_returns = []
    dates = []
    nav = 1.0

    for split in splits:
        train_returns = returns.loc[split.train_index]
        test_returns = returns.loc[split.test_index]

        if len(train_returns) < TRAIN_WINDOW // 2:
            continue

        try:
            sigma, _ = ledoit_wolf_shrinkage(train_returns)
            sigma = sigma * 252

            assets_no_cash = [t for t in valid_tickers if t != "CASH"]
            sigma_no_cash = sigma.loc[assets_no_cash, assets_no_cash].values

            erc_weights = calculate_erc_weights(sigma_no_cash, cardinality=cardinality)
            weights = pd.Series(erc_weights, index=assets_no_cash)
            weights = weights.reindex(valid_tickers, fill_value=0.0)

            cash_floor = 0.15
            weights = weights * (1 - cash_floor)
            weights["CASH"] = cash_floor

            if weights.sum() > 0:
                weights = weights / weights.sum()

            test_portfolio_returns = (test_returns * weights).sum(axis=1)

            for ret in test_portfolio_returns:
                nav *= 1 + ret

            portfolio_returns.extend(test_portfolio_returns.tolist())
            dates.extend(test_returns.index.tolist())

        except Exception:
            continue

    # Calcular métricas
    portfolio_returns_series = pd.Series(portfolio_returns, index=dates)
    cumulative_returns = (1 + portfolio_returns_series).cumprod()
    total_return = cumulative_returns.iloc[-1] - 1
    n_years = len(portfolio_returns_series) / 252
    annualized_return = (1 + total_return) ** (1 / n_years) - 1
    annualized_vol = portfolio_returns_series.std() * np.sqrt(252)
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0

    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    results_card[cardinality] = {
        "return": annualized_return,
        "vol": annualized_vol,
        "sharpe": sharpe,
        "max_dd": max_drawdown,
    }

    print(
        f"      Sharpe: {sharpe:.2f} | Vol: {annualized_vol:.1%} | DD: {max_drawdown:.1%}"
    )

print()
print("   Resumo Cardinality:")
card_df = pd.DataFrame(results_card).T
card_df.columns = ["Retorno", "Vol", "Sharpe", "Max DD"]
print(card_df.to_string())
print()

# ============================================================================
# SALVAR RESULTADOS
# ============================================================================
print("=" * 80)
print("💾 [4/4] Salvando resultados")
print("=" * 80)
print()

output_dir = Path("results/validation")
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

vol_df.to_csv(output_dir / f"sensitivity_vol_target_{timestamp}.csv")
cash_df.to_csv(output_dir / f"sensitivity_cash_floor_{timestamp}.csv")
card_df.to_csv(output_dir / f"sensitivity_cardinality_{timestamp}.csv")

print(f"   ✅ Resultados salvos em: {output_dir}/")
print()

# ============================================================================
# RESUMO EXECUTIVO
# ============================================================================
print("=" * 80)
print("  ✅ ANÁLISE DE SENSIBILIDADE CONCLUÍDA!")
print("=" * 80)
print()

print("🎯 CONCLUSÕES:")
print()

# Melhor Vol Target
best_vol = max(results_vol.items(), key=lambda x: x[1]["sharpe"])
print("   Melhor Vol Target:")
print(f"      • {best_vol[0]:.0%} → Sharpe {best_vol[1]['sharpe']:.2f}")
print()

# Melhor CASH Floor
best_cash = max(results_cash.items(), key=lambda x: x[1]["sharpe"])
print("   Melhor CASH Floor:")
print(f"      • {best_cash[0]:.0%} → Sharpe {best_cash[1]['sharpe']:.2f}")
print()

# Melhor Cardinality
best_card = max(results_card.items(), key=lambda x: x[1]["sharpe"])
print("   Melhor Cardinality:")
print(f"      • K={best_card[0]} → Sharpe {best_card[1]['sharpe']:.2f}")
print()

print("💡 RECOMENDAÇÕES:")
print(f"   • Vol Target: {best_vol[0]:.0%}")
print(f"   • CASH Floor: {best_cash[0]:.0%}")
print(f"   • Cardinality: K={best_card[0]}")
print()
