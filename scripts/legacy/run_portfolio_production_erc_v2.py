#!/usr/bin/env python
"""
Sistema de Produção ERC - Versão 2 (Calibrado)

Correções implementadas:
1. ✅ Vol target: 10-12% via bisection γ
2. ✅ Position caps: max 10% + group constraints
3. ✅ Turnover target: ≤12% via bisection η
4. ✅ Cardinalidade: K=15 via top-K + re-otimização
5. ✅ Triggers: sinais consistentes (CVaR e DD negativos)
6. ✅ Custos: 15 bps one-way (30 bps round-trip)
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from itau_quant.estimators.cov import ledoit_wolf_shrinkage
from production_logger import ProductionLogger
from production_monitor import should_fallback_to_1N

print("=" * 80)
print("  SISTEMA DE PRODUÇÃO ERC v2.0 (CALIBRADO)")
print("=" * 80)
print()

# ============================================================================
# 1. CONFIGURAÇÃO
# ============================================================================

VOL_TARGET = 0.11  # 11% aa (range: 10-12%)
VOL_TOLERANCE = 0.01  # ±1%

TURNOVER_TARGET = 0.12  # 12% mensal
TURNOVER_TOLERANCE = 0.01  # ±1%

MAX_POSITION = 0.10  # 10% por ativo
CARDINALITY_K = 15  # Número de ativos ativos

TRANSACTION_COST_BPS = 15  # 15 bps one-way (30 bps round-trip)
TRANSACTION_COST_DECIMAL = TRANSACTION_COST_BPS / 10000.0

ESTIMATION_WINDOW = 252  # 1 ano

# Group constraints
GROUPS = {
    "commodities": {
        "assets": ["DBC", "USO", "GLD", "SLV"],
        "max": 0.25,  # ≤25% total
    },
    "energy": {
        "assets": ["DBC", "USO"],
        "max": 0.20,  # ≤20% energia
    },
    "crypto": {
        "assets": ["IBIT", "ETHA"],
        "max": 0.12,  # ≤12% total
        "per_asset_max": 0.08,  # ≤8% por ativo
    },
    "us_equity": {
        "assets": ["SPY", "QQQ", "IWM", "VTV", "VUG"],
        "min": 0.25,  # ≥25%
        "max": 0.55,  # ≤55%
    },
    "treasuries": {
        "assets": ["IEF", "TLT", "SHY"],
        "max": 0.45,  # ≤45%
    },
}

# ============================================================================
# 2. CARREGAR DADOS
# ============================================================================

print("📥 Carregando dados...")
returns = pd.read_parquet("data/processed/returns_full.parquet")
print(f"   ✅ {len(returns)} dias, {len(returns.columns)} ativos")
print(f"   Período: {returns.index[0].date()} a {returns.index[-1].date()}")
print()

recent_returns = returns.tail(ESTIMATION_WINDOW)
valid_tickers = list(recent_returns.columns)

# Portfolio returns (proxy com equal-weight)
portfolio_returns = (returns * (1.0 / len(valid_tickers))).sum(axis=1)

# ============================================================================
# 3. TESTAR TRIGGERS DE FALLBACK
# ============================================================================

print("🚨 Verificando triggers de fallback...")
fallback_needed, trigger_status, metrics = should_fallback_to_1N(
    portfolio_returns,
    lookback_days=126,
    sharpe_threshold=0.0,  # Sharpe ≤ 0 → fallback
    cvar_threshold=-0.02,  # CVaR < -2% → fallback
    dd_threshold=-0.10,  # DD < -10% → fallback
    verbose=True,
)
print()

# ============================================================================
# 4. OTIMIZAR PORTFOLIO
# ============================================================================

print("⚙️  Otimizando portfolio...")

# Estimar covariância
cov, shrinkage = ledoit_wolf_shrinkage(recent_returns)
cov_annual = cov * 252

print(f"   Σ via Ledoit-Wolf (shrinkage: {shrinkage:.4f})")

if fallback_needed:
    print("   ⚠️  FALLBACK ATIVADO → Usando 1/N")
    weights = pd.Series(1.0 / len(valid_tickers), index=valid_tickers)
    strategy = "1/N"
    n_active = len(valid_tickers)
    n_effective = len(valid_tickers)
    portfolio_vol = np.sqrt(weights.values @ cov_annual.values @ weights.values)
    gamma_opt = None
    eta_opt = None
    turnover_realized = 0.0

else:
    print("   ✅ Triggers OK → Usando ERC Calibrado")

    # Pesos anteriores (ou equal-weight se primeiro rebalance)
    w_prev = np.ones(len(valid_tickers)) / len(valid_tickers)
    costs = np.full(len(valid_tickers), TRANSACTION_COST_DECIMAL)

    # NOVA ESTRATÉGIA: Primeiro enforcar cardinalidade, DEPOIS calibrar
    # (cardinalidade muda drasticamente a vol, então precisa calibrar no suporte fixo)

    # Passo 1: Resolver ERC unconstrained para selecionar top-K
    print(f"   📐 Selecionando top-{CARDINALITY_K} ativos...")
    from erc_calibrated import solve_erc_core
    w_unconstrained, _ = solve_erc_core(
        cov=cov_annual.values,
        w_prev=w_prev,
        gamma=1.0,  # Valor inicial razoável
        eta=0.0,
        costs=costs,
        w_max=MAX_POSITION,
        groups=GROUPS,
        asset_names=valid_tickers,
        verbose=False,
    )

    # Selecionar top-K
    top_k_indices = np.argsort(w_unconstrained)[-CARDINALITY_K:]
    support_mask = np.zeros(len(w_unconstrained), dtype=bool)
    support_mask[top_k_indices] = True
    active_tickers = [valid_tickers[i] for i in range(len(valid_tickers)) if support_mask[i]]
    print(f"      Top-{CARDINALITY_K}: {', '.join(active_tickers[:5])}...")

    # Passo 2: Calibrar γ NO SUPORTE FIXO para vol target
    print(f"   📐 Calibrando γ para vol target {VOL_TARGET:.1%} (suporte fixo)...")
    # Criar função wrapper que mantém suporte fixo
    def solve_with_fixed_support(cov, w_prev, gamma, eta, costs, w_max, groups, asset_names):
        w, status = solve_erc_core(
            cov, w_prev, gamma, eta, costs, w_max, groups, asset_names,
            support_mask=support_mask, verbose=False
        )
        return w, status

    # Bisection manual para γ
    lo_gamma, hi_gamma = 1e-3, 1e3
    for i in range(25):
        gamma_test = np.sqrt(lo_gamma * hi_gamma)
        w_test, _ = solve_with_fixed_support(
            cov_annual.values, w_prev, gamma_test, 0.0, costs,
            MAX_POSITION, GROUPS, valid_tickers
        )
        vol_test = np.sqrt(w_test @ cov_annual.values @ w_test)

        if abs(vol_test - VOL_TARGET) < VOL_TOLERANCE:
            break

        if vol_test > VOL_TARGET + VOL_TOLERANCE:
            hi_gamma = gamma_test
        else:
            lo_gamma = gamma_test

    gamma_opt = gamma_test
    vol_realized = vol_test
    print(f"      γ* = {gamma_opt:.6f}, vol = {vol_realized:.4f}")

    # Passo 3: Calibrar η NO SUPORTE FIXO para turnover target
    print(f"   📐 Calibrando η para turnover target {TURNOVER_TARGET:.1%} (suporte fixo)...")
    lo_eta, hi_eta = 1e-5, 5.0
    for i in range(20):
        eta_test = (lo_eta + hi_eta) / 2
        w_test, _ = solve_with_fixed_support(
            cov_annual.values, w_prev, gamma_opt, eta_test, costs,
            MAX_POSITION, GROUPS, valid_tickers
        )
        to_test = np.sum(np.abs(w_test - w_prev))

        if abs(to_test - TURNOVER_TARGET) < TURNOVER_TOLERANCE:
            break

        if to_test > TURNOVER_TARGET + TURNOVER_TOLERANCE:
            lo_eta = eta_test
        else:
            hi_eta = eta_test

    eta_opt = eta_test
    to_realized = to_test
    print(f"      η* = {eta_opt:.6f}, turnover = {to_realized:.4f}")

    # Passo 4: Solução final
    w_final, _ = solve_with_fixed_support(
        cov_annual.values, w_prev, gamma_opt, eta_opt, costs,
        MAX_POSITION, GROUPS, valid_tickers
    )
    n_active = int((w_final > 1e-4).sum())

    # Converter para Series
    weights = pd.Series(w_final, index=valid_tickers)
    strategy = "ERC"

    # Métricas finais
    herfindahl = (weights ** 2).sum()
    n_effective = 1.0 / herfindahl
    portfolio_vol = np.sqrt(weights.values @ cov_annual.values @ weights.values)
    turnover_realized = to_realized

print()
print("   ✅ Otimização concluída!")
print(f"      Estratégia: {strategy}")
print(f"      N_active: {n_active}")
print(f"      N_effective: {n_effective:.1f}")
print(f"      Vol ex-ante: {portfolio_vol:.2%}")
if strategy == "ERC":
    print(f"      γ* = {gamma_opt:.6f}")
    print(f"      η* = {eta_opt:.6f}")
print()

# ============================================================================
# 5. VALIDAR CONSTRAINTS
# ============================================================================

print("🔍 Validando constraints...")

# Check 1: Position caps
violations_pos = (weights > MAX_POSITION).sum()
max_pos = weights.max()
print(f"   Position caps (max {MAX_POSITION:.0%}): {max_pos:.2%} - {'✅ OK' if violations_pos == 0 else '❌ VIOLADO'}")

# Check 2: Vol target
vol_ok = abs(portfolio_vol - VOL_TARGET) <= VOL_TOLERANCE
print(f"   Vol target ({VOL_TARGET:.1%} ± {VOL_TOLERANCE:.1%}): {portfolio_vol:.2%} - {'✅ OK' if vol_ok else '⚠️  FORA'}")

# Check 3: Turnover (se ERC)
if strategy == "ERC":
    to_ok = turnover_realized <= TURNOVER_TARGET + TURNOVER_TOLERANCE
    print(f"   Turnover target (≤{TURNOVER_TARGET:.1%}): {turnover_realized:.2%} - {'✅ OK' if to_ok else '⚠️  EXCEDIDO'}")

# Check 4: Cardinality
card_ok = abs(n_active - CARDINALITY_K) <= 2  # ±2 ativos de tolerância
print(f"   Cardinality (K={CARDINALITY_K}): {n_active} ativos - {'✅ OK' if card_ok else '⚠️  FORA'}")

# Check 5: Group constraints (exemplo: commodities)
if strategy == "ERC":
    commodities = GROUPS["commodities"]["assets"]
    comm_weight = weights[[t for t in commodities if t in weights.index]].sum()
    comm_ok = comm_weight <= GROUPS["commodities"]["max"]
    print(f"   Commodities (≤{GROUPS['commodities']['max']:.0%}): {comm_weight:.2%} - {'✅ OK' if comm_ok else '❌ VIOLADO'}")

    crypto = GROUPS["crypto"]["assets"]
    crypto_weight = weights[[t for t in crypto if t in weights.index]].sum()
    crypto_ok = crypto_weight <= GROUPS["crypto"]["max"]
    print(f"   Crypto (≤{GROUPS['crypto']['max']:.0%}): {crypto_weight:.2%} - {'✅ OK' if crypto_ok else '❌ VIOLADO'}")

print()

# ============================================================================
# 6. LOGGING
# ============================================================================

print("💾 Salvando rebalance...")
logger = ProductionLogger(log_dir=Path("results/production"))

# Turnover e custo (vs equal-weight baseline)
previous_weights = pd.Series(1.0 / len(valid_tickers), index=valid_tickers)
turnover_vs_baseline = np.abs(weights - previous_weights).sum()
cost_bps = turnover_vs_baseline * TRANSACTION_COST_BPS

logger.log_rebalance(
    date=datetime.now(),
    weights=weights,
    strategy=strategy,
    turnover_realized=turnover_vs_baseline,
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
# 7. RESUMO
# ============================================================================

print("=" * 80)
print("  📊 PORTFOLIO OTIMIZADO (v2.0)")
print("=" * 80)
print()

print("Alocação (top 10):")
top_weights = weights.nlargest(10)
for ticker in top_weights.index:
    bar = "█" * int(weights[ticker] * 200)
    print(f"   {ticker:6s}: {weights[ticker]:6.2%} {bar}")

print()
print("💰 Custos:")
print(f"   Turnover: {turnover_vs_baseline:.2%}")
print(f"   Custo: {cost_bps:.1f} bps (@ {TRANSACTION_COST_BPS} bps one-way)")
print()

print("📈 Métricas de Risco (6M):")
print(f"   Sharpe: {metrics.sharpe_6m:.2f}")
print(f"   CVaR 95%: {metrics.cvar_95:.2%}")
print(f"   Max DD: {metrics.max_dd:.2%}")
print()

if fallback_needed:
    print("⚠️  ATENÇÃO: Fallback para 1/N está ativo")
    print(f"   Razão: {trigger_status}")
else:
    print("✅ Sistema operando com ERC calibrado")

print()
print("=" * 80)
print("  ✅ REBALANCE CONCLUÍDO")
print("=" * 80)
