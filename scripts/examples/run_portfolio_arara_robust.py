#!/usr/bin/env python
"""
PRISM-R - Portfolio Risk Intelligence System
Carteira ARARA ROBUSTA - Arara Quant Lab

Script ROBUSTO para otimização de portfolio com:
- Estimação robusta de retornos (Shrunk_50 - mais conservador)
- Limites realistas por classe de ativo
- Custos de transação e turnover no solver
- Universo corrigido (IBIT spot vs BITO futuros)
- Budget constraints FUNCIONANDO ✅

Agora usa arquivos YAML de configuração para flexibilidade.
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from arara_quant.config import PortfolioConfig, UniverseConfig, get_settings, load_config

SETTINGS = get_settings()

print("=" * 80)
print("  PRISM-R - Portfolio Risk Intelligence System")
print("  Carteira ARARA ROBUSTA - Otimização com Estimação Robusta")
print("=" * 80)
print()

# ============================================================================
# CONFIGURAÇÃO ROBUSTA
# ============================================================================

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run ARARA robust portfolio optimization")
parser.add_argument(
    "--universe",
    type=str,
    default="configs/universe_arara_robust.yaml",
    help="Path to universe config file",
)
parser.add_argument(
    "--portfolio",
    type=str,
    default="configs/portfolio_arara_robust.yaml",
    help="Path to portfolio config file",
)
args = parser.parse_args()

# Load configurations
try:
    universe_config = load_config(args.universe, UniverseConfig)
    portfolio_config = load_config(args.portfolio, PortfolioConfig)
except Exception as e:
    print(f"❌ Error loading configuration: {e}")
    sys.exit(1)

# Extract parameters from configs
TICKERS = universe_config.tickers
RISK_AVERSION = portfolio_config.risk_aversion
MAX_POSITION = portfolio_config.max_position
MIN_POSITION = portfolio_config.min_position
TURNOVER_PENALTY = portfolio_config.turnover_penalty
ESTIMATION_WINDOW = portfolio_config.estimation_window
SHRINKAGE_METHOD = portfolio_config.shrinkage_method

# Estimator parameters
if portfolio_config.estimators:
    SHRINK_STRENGTH = getattr(portfolio_config.estimators, "shrink_strength", 0.5)
else:
    SHRINK_STRENGTH = 0.5

# Data parameters
if portfolio_config.data:
    lookback_years = portfolio_config.data.lookback_years
    min_history_days = portfolio_config.data.min_history_days
else:
    lookback_years = 3
    min_history_days = 302

# Período de análise
END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=365 * lookback_years)

# Transaction costs (not in basic schema yet, using hardcoded default)
TRANSACTION_COST_BPS = 30  # 30 bps round-trip

# Risk budget limits (defaults can be overridden in config if present)
DEFAULT_CLASS_LIMITS = {
    "crypto": 0.10,
    "precious": 0.15,
    "commodities_all": 0.25,
    "china": 0.10,
    "us_equity_min": 0.30,
    "us_equity_max": 0.70,
}

CLASS_LIMITS = DEFAULT_CLASS_LIMITS.copy()
config_class_limits = getattr(portfolio_config, "class_limits", None)
if config_class_limits:
    if hasattr(config_class_limits, "model_dump"):
        CLASS_LIMITS.update(config_class_limits.model_dump())
    elif isinstance(config_class_limits, dict):
        CLASS_LIMITS.update(config_class_limits)

TURNOVER_CAP = getattr(portfolio_config, "turnover_cap", 0.15)

MAX_DOWNLOAD_RETRIES = 3
RETRY_SLEEP_SECONDS = 5


def _download_market_data(tickers, start, end):
    """Download market data with simple retry/backoff."""
    last_error: Exception | None = None
    for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
        try:
            data = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
            )
            if data is None or data.empty:
                raise RuntimeError("yfinance retornou DataFrame vazio.")

            if attempt > 1:
                print(f"   ✅ Download concluído na tentativa {attempt}.")
            return data
        except Exception as exc:  # noqa: BLE001 - log de erro específico
            last_error = exc
            print(
                f"   ⚠️  Falha na tentativa {attempt}/{MAX_DOWNLOAD_RETRIES}: {exc}"
            )
            if attempt < MAX_DOWNLOAD_RETRIES:
                print(f"   ⏳ Aguardando {RETRY_SLEEP_SECONDS}s antes de tentar novamente...")
                time.sleep(RETRY_SLEEP_SECONDS)

    raise RuntimeError("Falha ao baixar dados após múltiplas tentativas.") from last_error

print("📊 Configuração ROBUSTA:")
print(f"   • Universe: {universe_config.name} ({len(TICKERS)} ativos)")
print(f"   • Período: {START_DATE.date()} a {END_DATE.date()}")
print(f"   • Risk Aversion: {RISK_AVERSION}")
print(f"   • Max Position: {MAX_POSITION:.1%}")
print(f"   • Turnover Penalty: {TURNOVER_PENALTY}")
print(f"   • Transaction Costs: {TRANSACTION_COST_BPS} bps round-trip")
print(f"   • Window: {ESTIMATION_WINDOW} dias")
print(f"   • μ estimador: Shrunk_50 (strength={SHRINK_STRENGTH:.2f})")
print("   • Config files:")
print(f"      - Universe: {args.universe}")
print(f"      - Portfolio: {args.portfolio}")
print()

# ============================================================================
# 1. CARREGAR DADOS
# ============================================================================
print("📥 [1/6] Carregando dados do mercado...")

try:
    import yfinance as yf

    print(f"   Baixando dados de {len(TICKERS)} ativos...")
    data = _download_market_data(TICKERS, START_DATE, END_DATE)

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data

    prices = prices.dropna(how="all")
    prices = prices.ffill().bfill()

    # Filtrar ativos com dados suficientes
    min_obs = min_history_days
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
print("   ✅ Estatísticas:")
print(f"      • Média diária: {returns.mean().mean():.4%}")
print(f"      • Vol diária:   {returns.std().mean():.4%}")
print()

# ============================================================================
# 3. ESTIMAR PARÂMETROS COM ROBUSTEZ (μ, Σ)
# ============================================================================
print("📈 [3/6] Estimando parâmetros com métodos ROBUSTOS...")

from arara_quant.estimators.cov import ledoit_wolf_shrinkage
from arara_quant.estimators.mu import shrunk_mean

recent_returns = returns.tail(ESTIMATION_WINDOW)

# ESTIMAÇÃO ROBUSTA DE μ via Shrunk_50 (conforme PRD)
# Validação OOS: Shrunk_50 oferece Sharpe mais conservador e realista
print(f"   Estimando μ via Shrunk_50 (strength={SHRINK_STRENGTH:.2f})...")
mu_shrunk = shrunk_mean(recent_returns, strength=SHRINK_STRENGTH, prior=0.0)
mu_annual = mu_shrunk * 252

print("   ✅ Shrunk mean calculado (prior 0.0)")
print("      Nota: abordagem conservadora recomendada no relatório final")

# ESTIMAÇÃO DE Σ via Ledoit-Wolf
print("   Estimando Σ via Ledoit-Wolf shrinkage...")
sigma, shrinkage = ledoit_wolf_shrinkage(recent_returns)
sigma_annual = sigma * 252

print(f"   ✅ Ledoit-Wolf shrinkage: {shrinkage:.4f}")
print()

print("   ✅ Retornos esperados robustos (anualizados, top 5):")
top5 = mu_annual.nlargest(5)
for ticker in top5.index:
    print(f"      {ticker}: {mu_annual[ticker]:+.2%}")

print(f"   ✅ Covariância estimada: {sigma_annual.shape}")
print()

# ============================================================================
# 4. DEFINIR CONSTRAINTS POR CLASSE DE ATIVO
# ============================================================================
print("🔒 [4/6] Definindo constraints por classe de ativo...")

from arara_quant.risk.budgets import RiskBudget

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

from arara_quant.optimization.core.mv_qp import MeanVarianceConfig, solve_mean_variance

# Custos de transação
cost_vector = pd.Series(TRANSACTION_COST_BPS / 10000, index=valid_tickers)

# Configuração com budget constraints integrados
config = MeanVarianceConfig(
    risk_aversion=RISK_AVERSION,
    turnover_penalty=TURNOVER_PENALTY,
    turnover_cap=None,  # Defina, se desejado, um cap ex: 0.10 para limitar Δw
    lower_bounds=pd.Series(MIN_POSITION, index=valid_tickers),
    upper_bounds=pd.Series(MAX_POSITION, index=valid_tickers),
    previous_weights=pd.Series(0.0, index=valid_tickers),
    cost_vector=cost_vector,
    budgets=budgets,  # ← AGORA INTEGRADO AO SOLVER
    solver="CLARABEL",  # CLARABEL com tolerâncias estritas por default
)

try:
    result = solve_mean_variance(mu_annual, sigma_annual, config)

    print("   ✅ Otimização concluída!")
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

print("   ✅ Portfolio final:")
print(f"      • {n_active} ativos ativos (peso > 0.1%)")
print(f"      • Soma dos pesos: {weights.sum():.6f}")
print()

print("   📊 Alocação (top 10):")
for ticker in active_weights.head(10).index:
    w = weights[ticker]
    bar_length = int(w * 200)
    bar = "█" * bar_length
    print(f"      {ticker:6s}: {w:6.2%} {bar}")
print()

# Validar budgets manualmente (budget_slack retorna formato incompatível)
print("   🔍 Validação de Risk Budgets:")
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

print("   📈 Métricas Ex-Ante (anualizadas):")
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

print("   📊 Diversificação:")
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

print("   🎯 Exposição por classe de ativo:")
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

output_dir = SETTINGS.results_dir
output_dir.mkdir(parents=True, exist_ok=True)

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
    "mu_estimator": "shrunk_50",
    "shrink_strength": SHRINK_STRENGTH,
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
print("🎯 Comparação com versão original:")
print(f"   • Sharpe ex-ante: {sharpe:.2f} (vs ~2.15 original)")
print(f"   • N_effective: {effective_n:.1f} (vs ~7.4 original)")
print(f"   • Max position: {weights.max():.1%} (teto: {MAX_POSITION:.0%})")
print()
print("📁 Arquivos gerados:")
print(f"   • {weights_file}")
print(f"   • {metrics_file}")
print()
print("⚠️  PRÓXIMOS PASSOS CRÍTICOS:")
print("   1. Rodar walk-forward backtest (OOS validation)")
print("   2. Comparar com baselines (1/N, min-var, risk parity)")
print("   3. Verificar se Sharpe OOS ≥ Sharpe baseline + 0.2")
print(f"   4. Validar turnover realizado ≤ {TURNOVER_CAP:.0%}/mês")
print()
