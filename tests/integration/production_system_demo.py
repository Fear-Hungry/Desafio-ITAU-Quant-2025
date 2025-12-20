#!/usr/bin/env python
"""
Teste completo do sistema de produção usando dados locais
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from arara_quant.utils.production_logger import ProductionLogger
from arara_quant.utils.production_monitor import (
    should_fallback_to_1N,
)

print("=" * 80)
print("  TESTE COMPLETO DO SISTEMA DE PRODUÇÃO")
print("  Risk Parity (ERC) + Fallback Automático")
print("=" * 80)
print()

# Carregar dados salvos localmente
print("📥 Carregando dados salvos localmente...")
returns = pd.read_parquet("data/processed/returns_full.parquet")
print(f"   ✅ Dados carregados: {len(returns)} dias, {len(returns.columns)} ativos")
print(f"   Período: {returns.index[0].date()} a {returns.index[-1].date()}")
print()

# Usar últimos 252 dias para estimação
ESTIMATION_WINDOW = 252
MAX_POSITION = 0.10
TRANSACTION_COST_BPS = 30

recent_returns = returns.tail(ESTIMATION_WINDOW)
valid_tickers = list(recent_returns.columns)

print("📊 Configuração:")
print(f"   Ativos: {len(valid_tickers)}")
print(f"   Janela: {ESTIMATION_WINDOW} dias")
print(f"   Max pos: {MAX_POSITION:.0%}")
print()

# Calcular portfolio returns (equal-weight como proxy)
print("📊 Calculando portfolio returns...")
portfolio_returns = (returns * (1.0 / len(valid_tickers))).sum(axis=1)
print(f"   ✅ {len(portfolio_returns)} retornos calculados")
print()

# Testar triggers
print("🚨 Testando triggers de fallback...")
fallback_needed, trigger_status, metrics = should_fallback_to_1N(
    portfolio_returns,
    lookback_days=126,
    verbose=True,
)
print()

# Otimizar portfolio
print("⚙️  Otimizando portfolio...")

from arara_quant.estimators.cov import ledoit_wolf_shrinkage
from arara_quant.optimization.core.risk_parity import iterative_risk_parity

# Estimar covariância
cov, shrinkage = ledoit_wolf_shrinkage(recent_returns)
cov_annual = cov * 252

print(f"   Σ estimada via Ledoit-Wolf (shrinkage: {shrinkage:.4f})")

if fallback_needed:
    print("   ⚠️  FALLBACK ATIVADO → Usando 1/N")
    weights = pd.Series(1.0 / len(valid_tickers), index=valid_tickers)
    strategy = "1/N"
else:
    print("   ✅ Triggers OK → Usando ERC (Risk Parity)")
    weights = iterative_risk_parity(cov_annual)
    weights = weights.clip(0, MAX_POSITION)
    weights = weights / weights.sum()
    strategy = "ERC"

# Métricas
n_active = (weights > 0.001).sum()
herfindahl = (weights**2).sum()
n_effective = 1.0 / herfindahl
portfolio_vol = np.sqrt(weights.values @ cov_annual.values @ weights.values)

print("   ✅ Otimização concluída!")
print(f"      Estratégia: {strategy}")
print(f"      N_active: {n_active}")
print(f"      N_effective: {n_effective:.1f}")
print(f"      Vol ex-ante: {portfolio_vol:.2%}")
print()

# Logging
print("💾 Testando logging...")
logger = ProductionLogger(log_dir=Path("outputs/results/production"))

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

# Resumo
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
print("💰 Custos de Rebalance:")
print(f"   Turnover: {turnover_realized:.2%}")
print(f"   Custo: {cost_bps:.1f} bps")
print()

print("📈 Métricas de Risco (6M):")
print(f"   Sharpe: {metrics.sharpe_6m:.2f}")
print(f"   CVaR 95%: {metrics.cvar_95:.2%}")
print(f"   Max DD: {metrics.max_dd:.2%}")
print()

if fallback_needed:
    print("⚠️  ATENÇÃO: Fallback para 1/N está ativo")
else:
    print("✅ Sistema operando normalmente com ERC")

print()

# Ver histórico de logs
print("=" * 80)
print("  HISTÓRICO DE LOGS")
print("=" * 80)
print()

logger.print_summary(last_n=10)

print()
print("=" * 80)
print("  ✅ SISTEMA 100% FUNCIONAL!")
print("=" * 80)
print()
print("Próximos passos:")
print("1. Revisar pesos propostos")
print("2. Executar trades via broker")
print("3. Monitorar triggers diariamente")
print("4. Consultar RUNBOOK_PRODUCAO.md para procedimentos")
