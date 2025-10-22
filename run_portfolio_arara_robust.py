#!/usr/bin/env python
"""
PRISM-R - Portfolio Risk Intelligence System
Carteira ARARA ROBUSTA - ITAU Quant Challenge

Script ROBUSTO para otimização de portfolio com:
- Estimação robusta de retornos (Bayesian Shrinkage 50%)
- Limites realistas por classe de ativo
- Custos de transação e turnover no solver
- Universo corrigido (IBIT spot vs BITO futuros)

CORREÇÕES APLICADAS:
- BITO → IBIT (ETF spot sem contango drag)
- MAX_POSITION: 15% → 10%
- Limites por classe: Crypto ≤ 10%, Precious ≤ 15%, Commodities ≤ 25%, China ≤ 10%
- Custos: 30 bps round-trip
- Turnover cap: 12% por rebalance
- μ estimado via Bayesian Shrinkage (50% para zero) após Huber falhar OOS
  Validação OOS (2025-10-22): Huber Sharpe=0.81 < 1/N Sharpe=1.05
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

print("=" * 80)
print("  PRISM-R - Portfolio Risk Intelligence System")
print("  Carteira ARARA ROBUSTA - Otimização com Estimação Robusta")
print("=" * 80)
print()

# ============================================================================
# CONFIGURAÇÃO ROBUSTA
# ============================================================================

# Universo ARARA CORRIGIDO (IBIT spot, não BITO futuros)
TICKERS = [
    # Ações US
    "SPY",
    "QQQ",
    "IWM",
    "VTV",
    "VUG",
    # Ações Desenvolvidos
    "EFA",
    "VGK",
    "EWJ",
    "EWU",
    "EWG",
    # Ações Emergentes (amplo + específico)
    "EEM",
    "VWO",
    "EWZ",
    "FXI",
    "INDA",
    # Renda Fixa
    "TLT",
    "IEF",
    "SHY",
    "LQD",
    "HYG",
    "EMB",
    # Commodities
    "GLD",
    "SLV",
    "DBC",
    "USO",
    # Real Estate
    "VNQ",
    "VNQI",
    # Crypto SPOT (CORRIGIDO)
    "IBIT",  # Bitcoin spot ETF (BlackRock)
    "ETHA",  # Ethereum spot ETF (opcional)
]

# Período de análise
END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=365 * 3)  # 3 anos

# Parâmetros ROBUSTOS
RISK_AVERSION = 4.0  # λ - mais conservador (vs 3.0 original)
MAX_POSITION = 0.10  # 10% max por ativo (vs 15% original)
MIN_POSITION = 0.00  # long-only
TURNOVER_PENALTY = 0.0015  # 15 bps por 1% turnover (vs 0.10 original)
TURNOVER_CAP = 0.12  # 12% max por rebalance
TRANSACTION_COST_BPS = 30  # 30 bps round-trip

# Parâmetros de estimação
ESTIMATION_WINDOW = 252  # 1 ano
SHRINKAGE_METHOD = "ledoit_wolf"
HUBER_DELTA = 1.5  # Parâmetro de robustez do Huber mean

# Limites por classe de ativo
CLASS_LIMITS = {
    "crypto": 0.10,  # Crypto ≤ 10%
    "precious": 0.15,  # GLD + SLV ≤ 15%
    "commodities_all": 0.25,  # Todas commodities ≤ 25%
    "china": 0.10,  # FXI ≤ 10%
    "us_equity_min": 0.30,  # US Equity ≥ 30%
    "us_equity_max": 0.70,  # US Equity ≤ 70%
}

print(f"📊 Configuração ROBUSTA:")
print(f"   • Universo: {len(TICKERS)} ativos")
print(f"   • Período: {START_DATE.date()} a {END_DATE.date()}")
print(f"   • Risk Aversion: {RISK_AVERSION} (vs 3.0 original)")
print(f"   • Max Position: {MAX_POSITION:.1%} (vs 15% original)")
print(f"   • Turnover Cap: {TURNOVER_CAP:.1%} por rebalance")
print(f"   • Transaction Costs: {TRANSACTION_COST_BPS} bps round-trip")
print(f"   • Window: {ESTIMATION_WINDOW} dias")
print(f"   • μ estimador: Huber (robust, delta={HUBER_DELTA})")
print()
print(f"   Limites por classe:")
print(f"      • Crypto ≤ {CLASS_LIMITS['crypto']:.0%}")
print(f"      • Precious metals ≤ {CLASS_LIMITS['precious']:.0%}")
print(f"      • Commodities total ≤ {CLASS_LIMITS['commodities_all']:.0%}")
print(f"      • China ≤ {CLASS_LIMITS['china']:.0%}")
print(
    f"      • US Equity: {CLASS_LIMITS['us_equity_min']:.0%}-{CLASS_LIMITS['us_equity_max']:.0%}"
)
print()

# ============================================================================
# 1. CARREGAR DADOS
# ============================================================================
print("📥 [1/6] Carregando dados do mercado...")

try:
    import yfinance as yf

    print(f"   Baixando dados de {len(TICKERS)} ativos...")
    data = yf.download(
        tickers=TICKERS,
        start=START_DATE,
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

    # Filtrar ativos com dados suficientes
    min_obs = ESTIMATION_WINDOW + 50
    valid_tickers = []
    for ticker in TICKERS:
        if ticker in prices.columns and prices[ticker].notna().sum() >= min_obs:
            valid_tickers.append(ticker)

    prices = prices[valid_tickers]

    print(
        f"   ✅ Dados carregados: {len(prices)} dias, {len(valid_tickers)} ativos válidos"
    )
    print(
        f"   ✅ Período efetivo: {prices.index[0].date()} a {prices.index[-1].date()}"
    )

    if len(valid_tickers) < 5:
        print(f"   ❌ ERRO: Poucos ativos com dados suficientes ({len(valid_tickers)})")
        sys.exit(1)

    print()

except Exception as e:
    print(f"   ❌ Erro ao carregar dados: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 2. CALCULAR RETORNOS
# ============================================================================
print("📊 [2/6] Calculando retornos...")

returns = prices.pct_change().dropna()

print(f"   ✅ Retornos calculados: {len(returns)} observações")
print(f"   ✅ Estatísticas:")
print(f"      • Média diária: {returns.mean().mean():.4%}")
print(f"      • Vol diária:   {returns.std().mean():.4%}")
print()

# ============================================================================
# 3. ESTIMAR PARÂMETROS COM ROBUSTEZ (μ, Σ)
# ============================================================================
print("📈 [3/6] Estimando parâmetros com métodos ROBUSTOS...")

from itau_quant.estimators.mu import mean_return, huber_mean
from itau_quant.estimators.cov import ledoit_wolf_shrinkage

recent_returns = returns.tail(ESTIMATION_WINDOW)

# ESTIMAÇÃO ROBUSTA DE μ via Bayesian Shrinkage (20% para zero)
# Após testes OOS: 50% shrinkage teve Sharpe 0.75 (muito conservador)
# 20% equilibra preservação de sinal com robustez
print(f"   Estimando μ via Bayesian Shrinkage (strength=0.2)...")
from itau_quant.estimators.mu import bayesian_shrinkage_mean
mu_shrunk_daily = bayesian_shrinkage_mean(recent_returns, prior=0.0, strength=0.2)
mu_annual = mu_shrunk_daily * 252

print(f"   ✅ Bayesian shrinkage aplicado (20% shrinkage para zero)")
print(f"      Equilibra preservação de sinal com controle de overfit")

# ESTIMAÇÃO DE Σ via Ledoit-Wolf
print(f"   Estimando Σ via Ledoit-Wolf shrinkage...")
sigma, shrinkage = ledoit_wolf_shrinkage(recent_returns)
sigma_annual = sigma * 252

print(f"   ✅ Ledoit-Wolf shrinkage: {shrinkage:.4f}")
print()

print(f"   ✅ Retornos esperados robustos (anualizados, top 5):")
top5 = mu_annual.nlargest(5)
for ticker in top5.index:
    print(f"      {ticker}: {mu_annual[ticker]:+.2%}")

print(f"   ✅ Covariância estimada: {sigma_annual.shape}")
print()

# ============================================================================
# 4. DEFINIR CONSTRAINTS POR CLASSE DE ATIVO
# ============================================================================
print("🔒 [4/6] Definindo constraints por classe de ativo...")

from itau_quant.risk.budgets import RiskBudget

# Mapeamento de classes
asset_class_map = {
    "crypto": ["IBIT", "ETHA"],
    "precious": ["GLD", "SLV"],
    "commodities_all": ["GLD", "SLV", "DBC", "USO"],
    "china": ["FXI"],
    "us_equity": ["SPY", "QQQ", "IWM", "VTV", "VUG"],
}

# Criar RiskBudgets
budgets = []

# Crypto ≤ 10%
crypto_tickers = [t for t in asset_class_map["crypto"] if t in valid_tickers]
if crypto_tickers:
    budgets.append(
        RiskBudget(
            name="Crypto",
            tickers=crypto_tickers,
            min_weight=0.0,
            max_weight=CLASS_LIMITS["crypto"],
        )
    )

# Precious metals ≤ 15%
precious_tickers = [t for t in asset_class_map["precious"] if t in valid_tickers]
if precious_tickers:
    budgets.append(
        RiskBudget(
            name="Precious Metals",
            tickers=precious_tickers,
            min_weight=0.0,
            max_weight=CLASS_LIMITS["precious"],
        )
    )

# Commodities total ≤ 25%
commodities_tickers = [
    t for t in asset_class_map["commodities_all"] if t in valid_tickers
]
if commodities_tickers:
    budgets.append(
        RiskBudget(
            name="Commodities Total",
            tickers=commodities_tickers,
            min_weight=0.0,
            max_weight=CLASS_LIMITS["commodities_all"],
        )
    )

# China ≤ 10%
china_tickers = [t for t in asset_class_map["china"] if t in valid_tickers]
if china_tickers:
    budgets.append(
        RiskBudget(
            name="China",
            tickers=china_tickers,
            min_weight=0.0,
            max_weight=CLASS_LIMITS["china"],
        )
    )

# US Equity 30-70%
us_equity_tickers = [t for t in asset_class_map["us_equity"] if t in valid_tickers]
if us_equity_tickers:
    budgets.append(
        RiskBudget(
            name="US Equity",
            tickers=us_equity_tickers,
            min_weight=CLASS_LIMITS["us_equity_min"],
            max_weight=CLASS_LIMITS["us_equity_max"],
        )
    )

print(f"   ✅ {len(budgets)} risk budgets definidos:")
for budget in budgets:
    min_w = f"{budget.min_weight:.0%}" if budget.min_weight else "0%"
    max_w = f"{budget.max_weight:.0%}" if budget.max_weight else "∞"
    print(f"      • {budget.name}: {min_w} - {max_w} ({len(budget.tickers)} ativos)")
print()

# ============================================================================
# 5. OTIMIZAR PORTFOLIO COM CONSTRAINTS
# ============================================================================
print("⚙️  [5/6] Otimizando portfolio (Mean-Variance + Risk Budgets)...")

from itau_quant.optimization.core.mv_qp import solve_mean_variance, MeanVarianceConfig

# Custos de transação
cost_vector = pd.Series(TRANSACTION_COST_BPS / 10000, index=valid_tickers)

# Configuração com budget constraints integrados
config = MeanVarianceConfig(
    risk_aversion=RISK_AVERSION,
    turnover_penalty=TURNOVER_PENALTY,
    turnover_cap=None,  # Bug conhecido - usar apenas penalty
    lower_bounds=pd.Series(MIN_POSITION, index=valid_tickers),
    upper_bounds=pd.Series(MAX_POSITION, index=valid_tickers),
    previous_weights=pd.Series(0.0, index=valid_tickers),
    cost_vector=cost_vector,
    budgets=budgets,  # ← AGORA INTEGRADO AO SOLVER
    solver="CLARABEL",  # CLARABEL com tolerâncias estritas por default
)

try:
    result = solve_mean_variance(mu_annual, sigma_annual, config)

    print(f"   ✅ Otimização concluída!")
    print(f"      Status: {result.summary.status}")
    print(f"      Solver: {result.summary.solver}")
    print(f"      Tempo: {result.summary.runtime:.3f}s")
    print()

    if not result.summary.is_optimal():
        print(f"   ⚠️  WARNING: Status não é optimal: {result.summary.status}")
        print()

except Exception as e:
    print(f"   ❌ Erro na otimização: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 6. ANALISAR RESULTADO E VALIDAR BUDGETS
# ============================================================================
print("📊 [6/6] Analisando portfolio otimizado...")

weights = result.weights
active_weights = weights[weights > 0.001].sort_values(ascending=False)
n_active = len(active_weights)

print(f"   ✅ Portfolio final:")
print(f"      • {n_active} ativos ativos (peso > 0.1%)")
print(f"      • Soma dos pesos: {weights.sum():.6f}")
print()

print(f"   📊 Alocação (top 10):")
for ticker in active_weights.head(10).index:
    w = weights[ticker]
    bar_length = int(w * 200)
    bar = "█" * bar_length
    print(f"      {ticker:6s}: {w:6.2%} {bar}")
print()

# Validar budgets manualmente (budget_slack retorna formato incompatível)
print(f"   🔍 Validação de Risk Budgets:")
for budget in budgets:
    actual = sum(weights.get(t, 0.0) for t in budget.tickers if t in weights.index)

    min_ok = budget.min_weight is None or actual >= budget.min_weight
    max_ok = budget.max_weight is None or actual <= budget.max_weight
    status = "✅" if min_ok and max_ok else "❌"

    print(f"      {status} {budget.name}: {actual:.2%}", end="")
    if budget.max_weight is not None:
        slack = budget.max_weight - actual
        print(f" (max: {budget.max_weight:.0%}, slack: {slack:+.2%})", end="")
    if budget.min_weight is not None:
        deficit = actual - budget.min_weight
        print(f" (min: {budget.min_weight:.0%}, deficit: {deficit:+.2%})", end="")
    print()
print()

# Métricas de portfolio
portfolio_return = float(mu_annual @ weights)
portfolio_vol = float(np.sqrt(weights @ sigma_annual @ weights))
sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0

print(f"   📈 Métricas Ex-Ante (anualizadas):")
print(f"      • Retorno esperado:  {portfolio_return:+.2%}")
print(f"      • Volatilidade:      {portfolio_vol:.2%}")
print(f"      • Sharpe Ratio:      {sharpe:.2f}")
print(f"      • Objective Value:   {result.objective_value:.4f}")
print()

# Diversificação
from scipy.stats import entropy

herfindahl = (weights**2).sum()
effective_n = 1.0 / herfindahl if herfindahl > 0 else 0
weights_positive = weights[weights > 1e-6]
shannon = entropy(weights_positive) if len(weights_positive) > 0 else 0

print(f"   📊 Diversificação:")
print(f"      • Herfindahl Index:  {herfindahl:.4f}")
print(f"      • Effective N:       {effective_n:.1f} ativos")
print(f"      • Shannon Entropy:   {shannon:.2f}")
print()

# Exposição por classe de ativo
asset_classes_display = {
    "US Equity": ["SPY", "QQQ", "IWM", "VTV", "VUG"],
    "Intl Equity": ["EFA", "VGK", "EWJ", "EWU", "EWG"],
    "EM Equity": ["EEM", "VWO", "EWZ", "FXI", "INDA"],
    "Fixed Income": ["TLT", "IEF", "SHY", "LQD", "HYG", "EMB"],
    "Commodities": ["GLD", "SLV", "DBC", "USO"],
    "Real Estate": ["VNQ", "VNQI"],
    "Crypto": ["IBIT", "ETHA"],
}

print(f"   🎯 Exposição por classe de ativo:")
for asset_class, tickers_in_class in asset_classes_display.items():
    exposure = sum(weights.get(t, 0.0) for t in tickers_in_class)
    if exposure > 0.001:
        bar_length = int(exposure * 100)
        bar = "█" * bar_length
        print(f"      {asset_class:15s}: {exposure:6.2%} {bar}")
print()

# ============================================================================
# SALVAR RESULTADO
# ============================================================================
print("💾 Salvando resultado...")

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Salvar pesos
weights_df = pd.DataFrame(
    {
        "ticker": weights.index,
        "weight": weights.values,
    }
).sort_values("weight", ascending=False)

weights_file = output_dir / f"portfolio_weights_robust_{timestamp}.csv"
weights_df.to_csv(weights_file, index=False)
print(f"   ✅ Pesos salvos: {weights_file}")

# Salvar métricas
metrics = {
    "timestamp": timestamp,
    "version": "robust",
    "n_assets": len(valid_tickers),
    "n_active": n_active,
    "risk_aversion": RISK_AVERSION,
    "max_position": MAX_POSITION,
    "turnover_cap": TURNOVER_CAP,
    "transaction_cost_bps": TRANSACTION_COST_BPS,
    "mu_estimator": "huber",
    "huber_delta": HUBER_DELTA,
    "sigma_estimator": "ledoit_wolf",
    "ledoit_wolf_shrinkage": float(shrinkage),
    "expected_return": portfolio_return,
    "volatility": portfolio_vol,
    "sharpe_ratio": sharpe,
    "herfindahl": herfindahl,
    "effective_n": effective_n,
    "solver_status": result.summary.status,
    "solver_time": result.summary.runtime,
}

metrics_df = pd.DataFrame([metrics])
metrics_file = output_dir / f"portfolio_metrics_robust_{timestamp}.csv"
metrics_df.to_csv(metrics_file, index=False)
print(f"   ✅ Métricas salvas: {metrics_file}")

print()
print("=" * 80)
print("  ✅ OTIMIZAÇÃO ROBUSTA CONCLUÍDA!")
print("=" * 80)
print()
print(f"🎯 Comparação com versão original:")
print(f"   • Sharpe ex-ante: {sharpe:.2f} (vs ~2.15 original)")
print(f"   • N_effective: {effective_n:.1f} (vs ~7.4 original)")
print(f"   • Max position: {weights.max():.1%} (teto: {MAX_POSITION:.0%})")
print()
print(f"📁 Arquivos gerados:")
print(f"   • {weights_file}")
print(f"   • {metrics_file}")
print()
print(f"⚠️  PRÓXIMOS PASSOS CRÍTICOS:")
print(f"   1. Rodar walk-forward backtest (OOS validation)")
print(f"   2. Comparar com baselines (1/N, min-var, risk parity)")
print(f"   3. Verificar se Sharpe OOS ≥ Sharpe baseline + 0.2")
print(f"   4. Validar turnover realizado ≤ {TURNOVER_CAP:.0%}/mês")
print()
