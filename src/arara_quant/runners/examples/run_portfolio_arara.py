#!/usr/bin/env python
"""
PRISM-R - Portfolio Risk Intelligence System
Carteira ARARA - Arara Quant Lab

Script completo para rodar otimização de portfólio com dados reais.
Agora usa arquivos YAML de configuração para flexibilidade.
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from arara_quant.config import PortfolioConfig, UniverseConfig, get_settings, load_config

SETTINGS = get_settings()

print("=" * 80)
print("  PRISM-R - Portfolio Risk Intelligence System")
print("  Carteira ARARA - Otimização com Dados Reais")
print("=" * 80)
print()

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run ARARA portfolio optimization")
parser.add_argument(
    "--universe",
    type=str,
    default="configs/universe/universe_arara.yaml",
    help="Path to universe config file",
)
parser.add_argument(
    "--portfolio",
    type=str,
    default="configs/portfolio/portfolio_arara_basic.yaml",
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

print("📊 Configuração:")
print(f"   • Universe: {universe_config.name} ({len(TICKERS)} ativos)")
print(f"   • Período: {START_DATE.date()} a {END_DATE.date()}")
print(f"   • Risk Aversion: {RISK_AVERSION}")
print(f"   • Max Position: {MAX_POSITION:.1%}")
print(f"   • Window: {ESTIMATION_WINDOW} dias")
print("   • Config files:")
print(f"      - Universe: {args.universe}")
print(f"      - Portfolio: {args.portfolio}")
print()

# ============================================================================
# 1. CARREGAR DADOS
# ============================================================================
print("📥 [1/5] Carregando dados do mercado...")

try:
    import yfinance as yf

    # Download dados
    print(f"   Baixando dados de {len(TICKERS)} ativos...")
    data = yf.download(
        tickers=TICKERS,
        start=START_DATE,
        end=END_DATE,
        progress=False,
        auto_adjust=True,  # ajusta splits/dividendos
    )

    # Extrair apenas Adj Close
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data

    # Limpar dados
    prices = prices.dropna(how="all")  # remove dias sem nenhum dado

    # Preencher NaNs com forward fill (conservador)
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
print("📊 [2/5] Calculando retornos...")

# Retornos diários simples
returns = prices.pct_change().dropna()

print(f"   ✅ Retornos calculados: {len(returns)} observações")
print("   ✅ Estatísticas:")
print(f"      • Média diária: {returns.mean().mean():.4%}")
print(f"      • Vol diária:   {returns.std().mean():.4%}")
print()

# ============================================================================
# 3. ESTIMAR PARÂMETROS (μ, Σ)
# ============================================================================
print("📈 [3/5] Estimando parâmetros de risco/retorno...")

from arara_quant.estimators.cov import (
    ledoit_wolf_shrinkage,
    nonlinear_shrinkage,
    tyler_m_estimator,
)
from arara_quant.estimators.mu import mean_return

# Usar janela recente
recent_returns = returns.tail(ESTIMATION_WINDOW)

# Estimar retornos esperados
mu = mean_return(recent_returns, method="simple")
mu_annual = mu * 252  # anualizar

# Estimar covariância com shrinkage
print(f"   Método de shrinkage: {SHRINKAGE_METHOD}")

if SHRINKAGE_METHOD == "ledoit_wolf":
    sigma, shrinkage = ledoit_wolf_shrinkage(recent_returns)
    print(f"   Shrinkage parameter: {shrinkage:.4f}")
elif SHRINKAGE_METHOD == "nonlinear":
    sigma = nonlinear_shrinkage(recent_returns)
    print("   Nonlinear shrinkage aplicado")
elif SHRINKAGE_METHOD == "tyler":
    sigma = tyler_m_estimator(recent_returns)
    print("   Tyler M-estimator aplicado")
else:
    from arara_quant.estimators.cov import sample_cov

    sigma = sample_cov(recent_returns)
    print("   Sample covariance (sem shrinkage)")

sigma_annual = sigma * 252  # anualizar

print("   ✅ Retornos esperados (anualizados):")
top5 = mu_annual.nlargest(5)
for ticker in top5.index:
    print(f"      {ticker}: {mu_annual[ticker]:+.2%}")

print(f"   ✅ Covariância estimada: {sigma_annual.shape}")
print()

# ============================================================================
# 4. OTIMIZAR PORTFOLIO
# ============================================================================
print("⚙️  [4/5] Otimizando portfolio (Mean-Variance)...")

from arara_quant.optimization.core.mv_qp import MeanVarianceConfig, solve_mean_variance

# Configuração do otimizador
config = MeanVarianceConfig(
    risk_aversion=RISK_AVERSION,
    turnover_penalty=TURNOVER_PENALTY,
    turnover_cap=None,
    lower_bounds=pd.Series(MIN_POSITION, index=valid_tickers),
    upper_bounds=pd.Series(MAX_POSITION, index=valid_tickers),
    previous_weights=pd.Series(0.0, index=valid_tickers),  # sem posição prévia
    cost_vector=None,  # pode adicionar custos de transação
    solver="CLARABEL",
    solver_kwargs=None,
    risk_config=None,
    factor_loadings=None,
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
# 5. ANALISAR RESULTADO
# ============================================================================
print("📊 [5/5] Analisando portfolio otimizado...")

weights = result.weights

# Filtrar pesos significativos (> 0.1%)
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

# Índice Herfindahl (concentração)
herfindahl = (weights**2).sum()
# Effective N (1/HHI)
effective_n = 1.0 / herfindahl if herfindahl > 0 else 0

# Shannon Entropy (diversificação)
weights_positive = weights[weights > 1e-6]
shannon = entropy(weights_positive) if len(weights_positive) > 0 else 0

print("   📊 Diversificação:")
print(f"      • Herfindahl Index:  {herfindahl:.4f}")
print(f"      • Effective N:       {effective_n:.1f} ativos")
print(f"      • Shannon Entropy:   {shannon:.2f}")
print()

# Exposição por classe de ativo (simples)
asset_classes = {
    "US Equity": ["SPY", "QQQ", "IWM", "VTV", "VUG"],
    "Intl Equity": ["EFA", "VGK", "EWJ", "EWU", "EWG"],
    "EM Equity": ["VWO", "EWZ", "INDA", "EEM", "FXI"],
    "Fixed Income": ["TLT", "IEF", "SHY", "LQD", "HYG", "EMB"],
    "Commodities": ["GLD", "SLV", "DBC", "USO"],
    "Real Estate": ["VNQ", "VNQI"],
    "Crypto": ["IBIT", "ETHA"],
}

print("   🎯 Exposição por classe de ativo:")
for asset_class, tickers_in_class in asset_classes.items():
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

weights_file = output_dir / f"portfolio_weights_{timestamp}.csv"
weights_df.to_csv(weights_file, index=False)
print(f"   ✅ Pesos salvos: {weights_file}")

# Salvar métricas
metrics = {
    "timestamp": timestamp,
    "n_assets": len(valid_tickers),
    "n_active": n_active,
    "risk_aversion": RISK_AVERSION,
    "expected_return": portfolio_return,
    "volatility": portfolio_vol,
    "sharpe_ratio": sharpe,
    "herfindahl": herfindahl,
    "effective_n": effective_n,
    "solver_status": result.summary.status,
    "solver_time": result.summary.runtime,
}

metrics_df = pd.DataFrame([metrics])
metrics_file = output_dir / f"portfolio_metrics_{timestamp}.csv"
metrics_df.to_csv(metrics_file, index=False)
print(f"   ✅ Métricas salvas: {metrics_file}")

print()
print("=" * 80)
print("  ✅ OTIMIZAÇÃO CONCLUÍDA COM SUCESSO!")
print("=" * 80)
print()
print("🎯 Próximos passos:")
print(f"   1. Revisar alocação em: {weights_file}")
print(f"   2. Validar métricas em: {metrics_file}")
print("   3. Rodar backtest walk-forward para validação OOS")
print("   4. Comparar com benchmarks (SPY, 60/40, Risk Parity)")
print()
print("💡 Dicas:")
print("   • Ajuste RISK_AVERSION para controlar risco (2=agressivo, 5=conservador)")
print("   • Ajuste MAX_POSITION para controlar concentração")
print("   • Use TURNOVER_PENALTY para controlar custos de rebalanceamento")
print()
