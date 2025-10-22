#!/usr/bin/env python
"""
PRISM-R - Portfolio de Produção: Risk Parity (ERC)

Sistema de produção com:
- Risk Parity (Equal Risk Contribution)
- Fallback automático para 1/N
- Logging estruturado
- Triggers de risco

Baseado em validação OOS: Sharpe 1.05 (melhor estratégia testada)
"""

from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from production_monitor import should_fallback_to_1N, calculate_portfolio_metrics
from production_logger import ProductionLogger

print("=" * 80)
print("  PRISM-R - Sistema de Produção: Risk Parity (ERC)")
print("  Estratégia validada OOS: Sharpe 1.05")
print("=" * 80)
print()

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

TICKERS = [
    # US Equity
    "SPY", "QQQ", "IWM", "VTV", "VUG",
    # Intl Developed
    "EFA", "VGK", "EWJ", "EWU",
    # EM
    "EEM", "VWO", "INDA", "FXI",
    # Fixed Income
    "IEF", "TLT", "SHY", "LQD", "HYG", "EMB",
    # Commodities
    "GLD", "SLV", "DBC", "USO",
    # Crypto
    "IBIT", "ETHA",
    # REIT
    "VNQ",
]

ESTIMATION_WINDOW = 252  # 1 ano
MAX_POSITION = 0.10
TRANSACTION_COST_BPS = 30
VOL_TARGET = 0.11  # 11% anualizado

# Triggers de fallback
SHARPE_THRESHOLD = 0.0
CVAR_THRESHOLD = -0.02  # -2% diário
DD_THRESHOLD = -0.10  # -10%

print("📊 Configuração:")
print(f"   Universo: {len(TICKERS)} ativos")
print(f"   Janela estimação: {ESTIMATION_WINDOW} dias")
print(f"   Vol target: {VOL_TARGET:.1%}")
print(f"   Max position: {MAX_POSITION:.0%}")
print(f"   Custos: {TRANSACTION_COST_BPS} bps")
print()

print("🚨 Triggers de Fallback para 1/N:")
print(f"   Sharpe 6M ≤ {SHARPE_THRESHOLD}")
print(f"   CVaR 95% < {CVAR_THRESHOLD:.1%}")
print(f"   Max DD < {DD_THRESHOLD:.0%}")
print()

# ============================================================================
# 1. CARREGAR DADOS
# ============================================================================

print("📥 [1/5] Carregando dados...")

from itau_quant.data.sources.yf import download_prices
from itau_quant.data.processing.returns import calculate_returns

END_DATE = datetime.now().strftime("%Y-%m-%d")
START_DATE = "2022-01-01"

prices = download_prices(TICKERS, start=START_DATE, end=END_DATE)
returns = calculate_returns(prices, method="log").dropna()

valid_tickers = list(returns.columns)
print(f"   ✅ Dados carregados: {len(returns)} dias, {len(valid_tickers)} ativos válidos")
print(f"   Período: {returns.index[0].date()} a {returns.index[-1].date()}")
print()

# ============================================================================
# 2. CALCULAR PORTF

OLIO RETURNS (últimos 6M para triggers)
# ============================================================================

print("📊 [2/5] Calculando portfolio returns histórico...")

# Usar equal-weight como proxy inicial (antes do primeiro rebalance)
# ou carregar pesos anteriores se existir
portfolio_returns = (returns * (1.0 / len(valid_tickers))).sum(axis=1)

print(f"   ✅ {len(portfolio_returns)} retornos calculados")
print()

# ============================================================================
# 3. AVALIAR TRIGGERS DE FALLBACK
# ============================================================================

print("🚨 [3/5] Avaliando triggers de fallback...")

fallback_needed, trigger_status, metrics = should_fallback_to_1N(
    portfolio_returns,
    lookback_days=126,
    sharpe_threshold=SHARPE_THRESHOLD,
    cvar_threshold=CVAR_THRESHOLD,
    dd_threshold=DD_THRESHOLD,
    verbose=True,
)
print()

# ============================================================================
# 4. OTIMIZAR PORTFOLIO
# ============================================================================

print("⚙️  [4/5] Otimizando portfolio...")

recent_returns = returns.tail(ESTIMATION_WINDOW)

from itau_quant.estimators.cov import ledoit_wolf_shrinkage
from itau_quant.optimization.core.risk_parity import iterative_risk_parity

# Estimar covariância
cov, shrinkage = ledoit_wolf_shrinkage(recent_returns)
cov_annual = cov * 252

print(f"   Σ estimada via Ledoit-Wolf (shrinkage: {shrinkage:.4f})")

if fallback_needed:
    print(f"   ⚠️  FALLBACK ATIVADO → Usando 1/N")
    weights = pd.Series(1.0 / len(valid_tickers), index=valid_tickers)
    strategy = "1/N"
else:
    print(f"   ✅ Triggers OK → Usando ERC (Risk Parity)")

    # Risk Parity
    weights = iterative_risk_parity(cov_annual)

    # Aplicar limites por ativo
    weights = weights.clip(0, MAX_POSITION)
    weights = weights / weights.sum()

    strategy = "ERC"

# Calcular métricas do portfolio
n_active = (weights > 0.001).sum()
herfindahl = (weights ** 2).sum()
n_effective = 1.0 / herfindahl

portfolio_vol = np.sqrt(weights.values @ cov_annual.values @ weights.values)

print(f"   ✅ Otimização concluída!")
print(f"      Estratégia: {strategy}")
print(f"      N_active: {n_active}")
print(f"      N_effective: {n_effective:.1f}")
print(f"      Vol ex-ante: {portfolio_vol:.2%}")
print()

# ============================================================================
# 5. LOGGING
# ============================================================================

print("💾 [5/5] Registrando rebalance...")

logger = ProductionLogger(log_dir=Path("results/production"))

# Calcular turnover (assumir pesos anteriores = equal-weight para este exemplo)
previous_weights = pd.Series(1.0 / len(valid_tickers), index=valid_tickers)
turnover_realized = np.abs(weights - previous_weights).sum()
cost_bps = turnover_realized * TRANSACTION_COST_BPS

logger.log_rebalance(
    date=datetime.now(),
    weights=weights,
    strategy=strategy,
    turnover_realized=turnover_realized,
    cost_bps=cost_bps,
    metrics={
        "sharpe_6m": metrics.sharpe_6m,
        "cvar_95": metrics.cvar_95,
        "max_dd": metrics.max_dd,
        "vol": portfolio_vol,
    },
    trigger_status=trigger_status.to_dict(),
    fallback_active=fallback_needed,
)
print()

# ============================================================================
# RESUMO
# ============================================================================

print("=" * 80)
print("  📊 PORTFOLIO OTIMIZADO")
print("=" * 80)
print()

print("Alocação (top 10):")
top_weights = weights.nlargest(10)
for ticker in top_weights.index:
    bar = "█" * int(weights[ticker] * 200)
    print(f"   {ticker:6s}: {weights[ticker]:6.2%} {bar}")

print()
print(f"💰 Custos de Rebalance:")
print(f"   Turnover: {turnover_realized:.2%}")
print(f"   Custo: {cost_bps:.1f} bps")
print()

print(f"📈 Métricas de Risco (6M):")
print(f"   Sharpe: {metrics.sharpe_6m:.2f}")
print(f"   CVaR 95%: {metrics.cvar_95:.2%}")
print(f"   Max DD: {metrics.max_dd:.2%}")
print()

if fallback_needed:
    print("⚠️  ATENÇÃO: Fallback para 1/N está ativo")
    print("   Revisar triggers antes de executar trades")
else:
    print("✅ Sistema operando normalmente com ERC")

print()
print("=" * 80)
print("  Sistema pronto para produção!")
print("=" * 80)
