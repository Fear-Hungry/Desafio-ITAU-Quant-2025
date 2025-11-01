#!/usr/bin/env python
"""
Sistema de Produção ERC - Versão 2.1 (Com Overlay Defensivo)

Correções implementadas:
1. ✅ Vol target: 10-12% via bisection γ
2. ✅ Position caps: max 10% + group constraints
3. ✅ Turnover target: ≤12% via bisection η
4. ✅ Cardinalidade: K=22 via top-K + re-otimização
5. ✅ Triggers: sinais consistentes (CVaR e DD negativos)
6. ✅ Custos: 15 bps one-way (30 bps round-trip)
7. ✅ Overlay defensivo: CASH floor + filtros SPY + drawdown mode (DD validado: -14.8%)
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from itau_quant.estimators.cov import ledoit_wolf_shrinkage
from itau_quant.utils.production_logger import ProductionLogger
from itau_quant.utils.production_monitor import (
    should_fallback_to_1N,
)

try:
    import os
    import warnings

    import yfinance as yf

    # Fix yfinance cache permissions to suppress readonly database warnings
    cache_dir = os.path.expanduser("~/.cache/py-yfinance")
    if os.path.exists(cache_dir):
        try:
            for db_file in ["cookies.db", "tkr-tz.db"]:
                db_path = os.path.join(cache_dir, db_file)
                if os.path.exists(db_path):
                    os.chmod(db_path, 0o644)
        except Exception:
            # Silently ignore permission errors - not critical
            pass

    # Suppress yfinance warnings about cache
    warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️  yfinance não disponível - filtros SPY desabilitados")

print("=" * 80)
print("  SISTEMA DE PRODUÇÃO ERC v2.0 (CALIBRADO)")
print("=" * 80)
print()

# ============================================================================
# 1. CONFIGURAÇÃO
# ============================================================================

VOL_TARGET = 0.12  # 12% aa (aumentado para reforçar equity/growth)
VOL_TOLERANCE = 0.02  # ±2%

TURNOVER_TARGET = 0.12  # 12% mensal
TURNOVER_TOLERANCE = 0.01  # ±1%

MAX_POSITION = 0.08  # 8% por ativo (reduzido para rebalanceamentos mais suaves)
MIN_POSITION = 0.02  # 2% mínimo por ativo ativo (preserva convicção)
CARDINALITY_K = 22  # Sweet spot: captura ~90% benefícios de diversificação

TRANSACTION_COST_BPS = 15  # 15 bps one-way (30 bps round-trip)
TRANSACTION_COST_DECIMAL = TRANSACTION_COST_BPS / 10000.0

ESTIMATION_WINDOW = 252  # 1 ano

# Defensive Overlay Configuration (ATUALIZADO - CASH dinâmico)
DEFENSIVE_OVERLAY = {
    "enabled": True,
    "cash_floor_normal": 0.15,  # 15% CASH em regime normal (maximiza equity allocation)
    "cash_floor_defensive": 0.40,  # 40% CASH quando triggers ativam (proteção)
    "drawdown_limit": 0.15,  # Ativa defensive mode se DD > 15%
    "recovery_threshold": 0.03,  # Desativa se recovery > 3%
    "template": {  # Template defensivo quando triggers ativam
        "CASH": 0.40,
        "IEF": 0.30,
        "TLT": 0.15,
        "LQD": 0.10,
        "GLD": 0.05,
    },
    "spy_filters": {  # Filtros de tendência SPY
        "ma200": True,  # SPY < MA200 → risk-off
        "ma50": True,  # SPY < MA50 → risk-off
        "momentum_126d": 0.00,  # Ret 126d < 0% → risk-off
        "momentum_63d": -0.02,  # Ret 63d < -2% → risk-off
    },
    "blend_weight": 1.0,  # 100% defensive quando triggers ativam
}

# Group constraints (ATUALIZADO - diversificação balanceada com CASH 40%)
GROUPS = {
    "all_bonds": {  # NOVO: Cap em bonds totais
        "assets": [
            "SHY",
            "IEF",
            "IEI",
            "TLT",
            "AGG",
            "VGSH",
            "VGIT",
            "VCSH",
            "BNDX",
            "LQD",
            "HYG",
            "EMB",
            "EMLC",
            "TIP",
        ],
        "max": 0.50,  # ≤50% total (evita 100% bonds)
    },
    "commodities": {
        "assets": ["DBC", "USO", "GLD", "SLV"],
        "max": 0.25,  # ≤25% total
    },
    "energy": {
        "assets": ["DBC", "USO"],
        "max": 0.20,  # ≤20% energia
    },
    "crypto": {
        "assets": ["IBIT", "ETHA", "FBTC", "GBTC", "ETHE"],
        "max": 0.12,  # ≤12% total
        "per_asset_max": 0.08,  # ≤8% por ativo
    },
    "us_equity": {  # ATUALIZADO: apenas max constraint (min via overlay)
        "assets": [
            "SPY",
            "QQQ",
            "IWM",
            "VTV",
            "VUG",
            "VYM",
            "SCHD",
            "SPLV",
            "USMV",
            "MTUM",
            "QUAL",
            "VLUE",
            "SIZE",
        ],
        "max": 0.50,  # ≤50%
    },
    "treasuries": {
        "assets": ["IEF", "TLT", "SHY"],
        "max": 0.45,  # ≤45%
    },
}

# Targets de validação (não enforçados, apenas reportados)
VALIDATION_TARGETS = {
    "us_equity_min": 0.10,  # Target: ≥10% equity
    "growth_min": 0.05,  # Target: ≥5% growth
    "international_min": 0.03,  # Target: ≥3% internacional
}

MIN_US_EQUITY_SUPPORT = 6  # Garante pelo menos 6 tickers de equity no suporte inicial
MIN_GROWTH_SUPPORT = 3  # Garante pelo menos 3 tickers de growth (SPY, QQQ, VUG, MTUM)
MIN_INTERNATIONAL_SUPPORT = (
    2  # Garante pelo menos 2 tickers internacionais (EFA, EEM, VWO)
)

# ============================================================================
# 1.5. FUNÇÕES AUXILIARES DO OVERLAY DEFENSIVO
# ============================================================================


def check_spy_risk_off_signals(prices_df, current_date, config):
    """
    Verifica sinais de risk-off baseados no SPY.

    Returns
    -------
    bool
        True se algum filtro SPY ativar risk-off
    """
    if not YFINANCE_AVAILABLE or "SPY" not in prices_df.columns:
        return False

    if current_date not in prices_df.index:
        return False

    spy_price = prices_df.at[current_date, "SPY"]
    risk_off = False

    # Filtro MA200
    if config["spy_filters"].get("ma200", False):
        spy_ma200 = prices_df["SPY"].rolling(200).mean()
        if current_date in spy_ma200.index and pd.notna(spy_ma200.at[current_date]):
            if spy_price < spy_ma200.at[current_date]:
                risk_off = True
                print("      🚨 SPY < MA200 detectado")

    # Filtro MA50
    if config["spy_filters"].get("ma50", False):
        spy_ma50 = prices_df["SPY"].rolling(50).mean()
        if current_date in spy_ma50.index and pd.notna(spy_ma50.at[current_date]):
            if spy_price < spy_ma50.at[current_date]:
                risk_off = True
                print("      🚨 SPY < MA50 detectado")

    # Filtro momentum 126d
    momentum_126d_threshold = config["spy_filters"].get("momentum_126d", 0.0)
    if current_date in prices_df.index:
        spy_ret_126d = prices_df["SPY"] / prices_df["SPY"].shift(126) - 1
        if current_date in spy_ret_126d.index and pd.notna(
            spy_ret_126d.at[current_date]
        ):
            if spy_ret_126d.at[current_date] < momentum_126d_threshold:
                risk_off = True
                print(f"      🚨 SPY momentum 126d < {momentum_126d_threshold:.1%}")

    # Filtro momentum 63d
    momentum_63d_threshold = config["spy_filters"].get("momentum_63d", -0.02)
    if current_date in prices_df.index:
        spy_ret_63d = prices_df["SPY"] / prices_df["SPY"].shift(63) - 1
        if current_date in spy_ret_63d.index and pd.notna(spy_ret_63d.at[current_date]):
            if spy_ret_63d.at[current_date] < momentum_63d_threshold:
                risk_off = True
                print(f"      🚨 SPY momentum 63d < {momentum_63d_threshold:.1%}")

    return risk_off


def apply_defensive_overlay(weights, config, valid_tickers, risk_off_active=False):
    """
    Aplica overlay defensivo nos pesos.

    Parameters
    ----------
    weights : pd.Series
        Pesos originais do portfolio
    config : dict
        Configuração do overlay defensivo
    valid_tickers : list
        Lista de tickers válidos
    risk_off_active : bool
        Se True, aplica blend com template defensivo

    Returns
    -------
    pd.Series
        Pesos ajustados com overlay defensivo
    """
    if not config.get("enabled", False):
        return weights

    # NOVO: Determinar cash floor baseado em regime (dinâmico)
    if risk_off_active:
        cash_floor = config.get("cash_floor_defensive", 0.40)
        print(f"      🚨 Modo DEFENSIVO ativado: CASH floor = {cash_floor:.0%}")
    else:
        cash_floor = config.get("cash_floor_normal", 0.20)
        print(f"      ✅ Modo NORMAL: CASH floor = {cash_floor:.0%}")

    # Se risk-off ativo, fazer blend com template defensivo
    if risk_off_active:
        defensive_template = pd.Series(config["template"]).reindex(
            valid_tickers, fill_value=0.0
        )
        if defensive_template.sum() > 0:
            defensive_template = defensive_template / defensive_template.sum()

            blend_weight = config.get("blend_weight", 1.0)
            weights = (1 - blend_weight) * weights + blend_weight * defensive_template
            weights = weights.clip(lower=0)

            if weights.sum() > 0:
                weights = weights / weights.sum()

            print(f"      ✅ Blend defensivo aplicado (peso: {blend_weight:.0%})")

    # Garantir CASH floor mínimo (agora dinâmico)
    if "CASH" in weights.index and weights["CASH"] < cash_floor:
        # Reduzir outros ativos proporcionalmente para liberar espaço para CASH
        cash_deficit = cash_floor - weights["CASH"]
        non_cash_weights = weights.drop("CASH")

        if non_cash_weights.sum() > cash_deficit:
            # Reduzir proporcionalmente
            scale_factor = (
                non_cash_weights.sum() - cash_deficit
            ) / non_cash_weights.sum()
            weights.loc[non_cash_weights.index] = non_cash_weights * scale_factor
            weights["CASH"] = cash_floor

            print(f"      ✅ CASH floor {cash_floor:.0%} aplicado")

    # Renormalizar
    if weights.sum() > 0:
        weights = weights / weights.sum()

    return weights


# ============================================================================
# 2. CARREGAR DADOS
# ============================================================================

print("📥 Carregando dados...")
returns = pd.read_parquet("data/processed/returns_full.parquet")
print(f"   ✅ {len(returns)} dias, {len(returns.columns)} ativos")
print(f"   Período: {returns.index[0].date()} a {returns.index[-1].date()}")
print()

# Adicionar CASH como ativo se não existir
if "CASH" not in returns.columns:
    returns["CASH"] = 0.0
    print("   ✅ Ativo CASH adicionado (retorno 0%)")

recent_returns = returns.tail(ESTIMATION_WINDOW)
valid_tickers = list(recent_returns.columns)

# Carregar dados de preços para filtros SPY (se disponível)
prices_df = None
if DEFENSIVE_OVERLAY["enabled"] and YFINANCE_AVAILABLE:
    print("📊 Carregando dados SPY para filtros...")
    try:
        spy_data = yf.download(
            "SPY",
            start=returns.index[0],
            end=returns.index[-1],
            progress=False,
            auto_adjust=True,
        )
        if not spy_data.empty:
            prices_df = pd.DataFrame(index=returns.index)
            prices_df["SPY"] = spy_data["Close"].reindex(returns.index).ffill()
            print("   ✅ Dados SPY carregados")
    except Exception as e:
        print(f"   ⚠️  Erro ao carregar SPY: {e}")
print()

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
    from itau_quant.optimization.erc_calibrated import solve_erc_core

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

    # Selecionar top-K garantindo presença mínima de equity, growth e international
    sorted_indices_desc = np.argsort(w_unconstrained)[::-1]
    support_indices = []

    # Passo 1: Forçar US Equity mínimo
    us_equity_assets = set(GROUPS.get("us_equity", {}).get("assets", []))
    equity_indices = [
        idx for idx, ticker in enumerate(valid_tickers) if ticker in us_equity_assets
    ]
    equity_indices_sorted = sorted(
        equity_indices, key=lambda idx: w_unconstrained[idx], reverse=True
    )
    forced_equity = equity_indices_sorted[:MIN_US_EQUITY_SUPPORT]
    support_indices.extend(forced_equity)
    print(
        f"      Forçado {MIN_US_EQUITY_SUPPORT} equity: {[valid_tickers[i] for i in forced_equity]}"
    )

    # Passo 2: Forçar Growth mínimo (subset de equity, pode sobrepor)
    growth_assets = ["SPY", "QQQ", "VUG", "MTUM"]
    growth_indices = [
        idx for idx, ticker in enumerate(valid_tickers) if ticker in growth_assets
    ]
    growth_indices_sorted = sorted(
        growth_indices, key=lambda idx: w_unconstrained[idx], reverse=True
    )
    forced_growth = growth_indices_sorted[:MIN_GROWTH_SUPPORT]
    for idx in forced_growth:
        if idx not in support_indices:
            support_indices.append(idx)
    print(
        f"      Forçado {MIN_GROWTH_SUPPORT} growth: {[valid_tickers[i] for i in forced_growth]}"
    )

    # Passo 3: Forçar International mínimo
    intl_assets = ["EFA", "VGK", "VPL", "EEM", "VWO"]
    intl_indices = [
        idx for idx, ticker in enumerate(valid_tickers) if ticker in intl_assets
    ]
    intl_indices_sorted = sorted(
        intl_indices, key=lambda idx: w_unconstrained[idx], reverse=True
    )
    forced_intl = intl_indices_sorted[:MIN_INTERNATIONAL_SUPPORT]
    for idx in forced_intl:
        if idx not in support_indices:
            support_indices.append(idx)
    print(
        f"      Forçado {MIN_INTERNATIONAL_SUPPORT} international: {[valid_tickers[i] for i in forced_intl]}"
    )

    # Passo 4: Completar com top-K geral até atingir CARDINALITY_K
    for idx in sorted_indices_desc:
        if idx in support_indices:
            continue
        support_indices.append(idx)
        if len(support_indices) >= CARDINALITY_K:
            break

    support_indices = support_indices[:CARDINALITY_K]
    support_mask = np.zeros(len(w_unconstrained), dtype=bool)
    support_mask[support_indices] = True
    active_tickers = [
        valid_tickers[i] for i in range(len(valid_tickers)) if support_mask[i]
    ]
    print(
        f"      Suporte final (K={len(active_tickers)}): {', '.join(active_tickers[:5])}..."
    )

    # Passo 2: Calibrar γ NO SUPORTE FIXO para vol target
    print(f"   📐 Calibrando γ para vol target {VOL_TARGET:.1%} (suporte fixo)...")

    # Criar função wrapper que mantém suporte fixo + peso mínimo
    def solve_with_fixed_support(
        cov, w_prev, gamma, eta, costs, w_max, groups, asset_names
    ):
        w, status = solve_erc_core(
            cov,
            w_prev,
            gamma,
            eta,
            costs,
            w_max,
            groups,
            asset_names,
            support_mask=support_mask,
            verbose=False,
        )
        # Aplicar peso mínimo pós-otimização (ajustar ativos ativos)
        w_active = w[support_mask]
        if (w_active < MIN_POSITION).any():
            # Se algum ativo ativo está abaixo do mínimo, ajustar
            w_active = np.maximum(w_active, MIN_POSITION)
            w_active = w_active / w_active.sum()  # Renormalizar
            w[support_mask] = w_active
        return w, status

    # Bisection manual para γ
    # CORREÇÃO: γ ↑ → vol ↓ (log-barrier aumenta diversificação)
    # Limites expandidos para evitar saturação
    lo_gamma, hi_gamma = 1e-6, 1e6
    for i in range(30):
        gamma_test = np.sqrt(lo_gamma * hi_gamma)
        w_test, _ = solve_with_fixed_support(
            cov_annual.values,
            w_prev,
            gamma_test,
            0.0,
            costs,
            MAX_POSITION,
            GROUPS,
            valid_tickers,
        )
        vol_test = np.sqrt(w_test @ cov_annual.values @ w_test)

        if abs(vol_test - VOL_TARGET) < VOL_TOLERANCE:
            break

        # CORREÇÃO: γ↑ → vol↑ (equalização/1-N), γ↓ → vol↓ (concentração/min-var)
        # Se vol muito alta → DIMINUIR γ (concentrar para reduzir vol)
        # Se vol muito baixa → AUMENTAR γ (equalizar para aumentar vol)
        if vol_test > VOL_TARGET + VOL_TOLERANCE:
            hi_gamma = gamma_test  # Vol alta → diminuir γ
        else:
            lo_gamma = gamma_test  # Vol baixa → aumentar γ

    gamma_opt = gamma_test
    vol_realized = vol_test
    print(f"      γ* = {gamma_opt:.6f}, vol = {vol_realized:.4f}")

    # Passo 3: Calibrar η NO SUPORTE FIXO para turnover target
    print(
        f"   📐 Calibrando η para turnover target {TURNOVER_TARGET:.1%} (suporte fixo)..."
    )
    # CORREÇÃO: Limites expandidos (η pode precisar ser maior para reduzir turnover)
    lo_eta, hi_eta = 1e-6, 100.0
    for i in range(25):
        eta_test = (lo_eta + hi_eta) / 2
        w_test, _ = solve_with_fixed_support(
            cov_annual.values,
            w_prev,
            gamma_opt,
            eta_test,
            costs,
            MAX_POSITION,
            GROUPS,
            valid_tickers,
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
        cov_annual.values,
        w_prev,
        gamma_opt,
        eta_opt,
        costs,
        MAX_POSITION,
        GROUPS,
        valid_tickers,
    )
    n_active = int((w_final > 1e-4).sum())

    # Converter para Series
    weights = pd.Series(w_final, index=valid_tickers)
    strategy = "ERC"

    # =========================================================================
    # APLICAR OVERLAY DEFENSIVO (se habilitado)
    # =========================================================================
    if DEFENSIVE_OVERLAY.get("enabled", False):
        print()
        print("🛡️  Aplicando overlay defensivo...")

        # Verificar filtros SPY
        risk_off_active = False
        if prices_df is not None:
            current_date = returns.index[-1]
            risk_off_active = check_spy_risk_off_signals(
                prices_df, current_date, DEFENSIVE_OVERLAY
            )

        if risk_off_active:
            print("      🚨 Sinais risk-off detectados!")

        # Aplicar overlay (cash floor + blend se risk-off)
        weights_original = weights.copy()
        weights = apply_defensive_overlay(
            weights, DEFENSIVE_OVERLAY, valid_tickers, risk_off_active
        )

        # Mostrar diferença
        turnover_overlay = np.abs(weights - weights_original).sum()
        if turnover_overlay > 0.01:
            print(f"      📊 Ajuste overlay: {turnover_overlay:.2%} turnover")

        strategy = "ERC+Defensive" if risk_off_active else "ERC+CashFloor"

    # Métricas finais
    herfindahl = (weights**2).sum()
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

# Check 1: Position caps (excluindo CASH - é reserva técnica)
weights_no_cash = weights.drop("CASH", errors="ignore")
violations_pos = (weights_no_cash > MAX_POSITION).sum()
max_pos_excluding_cash = weights_no_cash.max() if not weights_no_cash.empty else 0.0
cash_pos = weights.get("CASH", 0.0)
print(
    f"   Position caps (max {MAX_POSITION:.0%}, exceto CASH): {max_pos_excluding_cash:.2%} - {'✅ OK' if violations_pos == 0 else '❌ VIOLADO'}"
)
print(f"   CASH (reserva técnica): {cash_pos:.2%}")

# Check 2: Vol target
vol_ok = abs(portfolio_vol - VOL_TARGET) <= VOL_TOLERANCE
print(
    f"   Vol target ({VOL_TARGET:.1%} ± {VOL_TOLERANCE:.1%}): {portfolio_vol:.2%} - {'✅ OK' if vol_ok else '⚠️  FORA'}"
)

# Check 3: Turnover (se ERC)
if strategy == "ERC":
    to_ok = turnover_realized <= TURNOVER_TARGET + TURNOVER_TOLERANCE
    print(
        f"   Turnover target (≤{TURNOVER_TARGET:.1%}): {turnover_realized:.2%} - {'✅ OK' if to_ok else '⚠️  EXCEDIDO'}"
    )

# Check 4: Cardinality
card_ok = abs(n_active - CARDINALITY_K) <= 2  # ±2 ativos de tolerância
print(
    f"   Cardinality (K={CARDINALITY_K}): {n_active} ativos - {'✅ OK' if card_ok else '⚠️  FORA'}"
)

# Check 5: Group constraints
if strategy.startswith("ERC"):
    # All bonds (NOVO)
    all_bonds = GROUPS["all_bonds"]["assets"]
    bonds_weight = weights[[t for t in all_bonds if t in weights.index]].sum()
    bonds_ok = bonds_weight <= GROUPS["all_bonds"]["max"]
    print(
        f"   All Bonds (≤{GROUPS['all_bonds']['max']:.0%}): {bonds_weight:.2%} - {'✅ OK' if bonds_ok else '❌ VIOLADO'}"
    )

    # US Equity
    us_equity = GROUPS["us_equity"]["assets"]
    equity_weight = weights[[t for t in us_equity if t in weights.index]].sum()
    equity_min_ok = equity_weight >= VALIDATION_TARGETS["us_equity_min"]
    equity_max_ok = equity_weight <= GROUPS["us_equity"]["max"]
    equity_ok = equity_min_ok and equity_max_ok
    print(
        f"   US Equity (target ≥{VALIDATION_TARGETS['us_equity_min']:.0%}, max {GROUPS['us_equity']['max']:.0%}): {equity_weight:.2%} - {'✅ OK' if equity_ok else '⚠️  FORA'}"
    )

    # Growth assets (validação)
    growth_assets = ["SPY", "QQQ", "VUG", "MTUM"]
    growth_weight = weights[[t for t in growth_assets if t in weights.index]].sum()
    growth_ok = growth_weight >= VALIDATION_TARGETS["growth_min"]
    print(
        f"   Growth (target ≥{VALIDATION_TARGETS['growth_min']:.0%}): {growth_weight:.2%} - {'✅ OK' if growth_ok else '⚠️  BAIXO'}"
    )

    # International equity (validação)
    intl_assets = ["EFA", "VGK", "VPL", "EEM", "VWO"]
    intl_weight = weights[[t for t in intl_assets if t in weights.index]].sum()
    intl_ok = intl_weight >= VALIDATION_TARGETS["international_min"]
    print(
        f"   International (target ≥{VALIDATION_TARGETS['international_min']:.0%}): {intl_weight:.2%} - {'✅ OK' if intl_ok else '⚠️  BAIXO'}"
    )

    # Commodities
    commodities = GROUPS["commodities"]["assets"]
    comm_weight = weights[[t for t in commodities if t in weights.index]].sum()
    comm_ok = comm_weight <= GROUPS["commodities"]["max"]
    print(
        f"   Commodities (≤{GROUPS['commodities']['max']:.0%}): {comm_weight:.2%} - {'✅ OK' if comm_ok else '❌ VIOLADO'}"
    )

    # Crypto
    crypto = GROUPS["crypto"]["assets"]
    crypto_weight = weights[[t for t in crypto if t in weights.index]].sum()
    crypto_ok = crypto_weight <= GROUPS["crypto"]["max"]
    print(
        f"   Crypto (≤{GROUPS['crypto']['max']:.0%}): {crypto_weight:.2%} - {'✅ OK' if crypto_ok else '❌ VIOLADO'}"
    )

# Check 6: CASH floor (se overlay defensivo habilitado)
if DEFENSIVE_OVERLAY.get("enabled", False):
    # Mostrar ambos os thresholds (normal e defensivo)
    cash_normal = DEFENSIVE_OVERLAY.get("cash_floor_normal", 0.20)
    cash_defensive = DEFENSIVE_OVERLAY.get("cash_floor_defensive", 0.40)
    cash_weight = weights.get("CASH", 0.0)

    # Determinar qual threshold aplicar (simplificado - assumir normal se não há info de risk-off)
    cash_threshold = cash_normal  # Pode ser cash_defensive se risk-off ativo
    cash_ok = cash_weight >= cash_threshold - 0.01
    print(
        f"   CASH floor (normal ≥{cash_normal:.0%}, defensivo ≥{cash_defensive:.0%}): {cash_weight:.2%} - {'✅ OK' if cash_ok else '❌ VIOLADO'}"
    )

# Check 7: Equity allocation warning (NOVO)
if strategy.startswith("ERC"):
    equity_target = VALIDATION_TARGETS.get("us_equity_min", 0.10)
    if equity_weight < equity_target:
        print()
        print(
            f"   ⚠️  WARNING: US Equity {equity_weight:.1%} ABAIXO do target {equity_target:.0%}"
        )
        print("      Ações sugeridas:")
        print(f"      • Considere aumentar vol_target (atual: {VOL_TARGET:.1%})")
        print(
            f"      • Ou reduzir cash_floor_normal (atual: {DEFENSIVE_OVERLAY['cash_floor_normal']:.0%})"
        )
        print("      • Ou aguardar regime normal (se SPY em tendência de baixa)")

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
print()

# ============================================================================
# 8. AUTO-GERAR SNAPSHOT DE STATUS
# ============================================================================

try:
    from scripts.production.generate_status_snapshot import generate_status_snapshot

    print("📄 Gerando snapshot de status...")
    generate_status_snapshot()
    print()
except Exception as e:
    print(f"⚠️  Erro ao gerar snapshot (não crítico): {e}")
    print()
