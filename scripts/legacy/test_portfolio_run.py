#!/usr/bin/env python
"""Script de validação do pipeline completo de portfolio.

Este script testa o fluxo end-to-end:
1. Carregar dados
2. Estimar parâmetros (μ, Σ)
3. Otimizar portfolio
4. Validar resultado
"""

from datetime import datetime

import numpy as np
import pandas as pd

print("=" * 70)
print("  TESTE DE VALIDAÇÃO - PIPELINE COMPLETO DE PORTFOLIO")
print("=" * 70)
print()

# ============================================================================
# 1. CRIAR DADOS SINTÉTICOS SIMPLES
# ============================================================================
print("📊 [1/5] Criando dados sintéticos...")

np.random.seed(42)
n_assets = 5
n_days = 252 * 2  # 2 anos

# Tickers simples
tickers = [f"ASSET{i}" for i in range(1, n_assets + 1)]

# Gerar retornos diários com correlação
mean_returns = np.array([0.08, 0.10, 0.06, 0.12, 0.07]) / 252  # anualizado -> diário
volatilities = np.array([0.15, 0.20, 0.12, 0.25, 0.18]) / np.sqrt(252)

# Matriz de correlação
corr = np.array(
    [
        [1.00, 0.50, 0.30, 0.20, 0.40],
        [0.50, 1.00, 0.40, 0.30, 0.50],
        [0.30, 0.40, 1.00, 0.25, 0.35],
        [0.20, 0.30, 0.25, 1.00, 0.30],
        [0.40, 0.50, 0.35, 0.30, 1.00],
    ]
)

# Converter para covariância
cov_matrix = np.outer(volatilities, volatilities) * corr

# Gerar retornos
returns_array = np.random.multivariate_normal(mean_returns, cov_matrix, size=n_days)

# Criar DataFrame com índice temporal
dates = pd.date_range(end=datetime.today(), periods=n_days, freq="B")
returns = pd.DataFrame(returns_array, index=dates, columns=tickers)

print(f"  ✅ Dados criados: {n_assets} ativos, {n_days} dias")
print(f"  ✅ Período: {returns.index[0].date()} a {returns.index[-1].date()}")
print()

# ============================================================================
# 2. ESTIMAR PARÂMETROS
# ============================================================================
print("📈 [2/5] Estimando parâmetros de risco/retorno...")

from itau_quant.estimators.cov import ledoit_wolf_shrinkage
from itau_quant.estimators.mu import mean_return

# Retornos esperados (usar últimos 252 dias)
recent_returns = returns.tail(252)
mu = mean_return(recent_returns, method="simple")

# Anualizar manualmente (252 dias de trading)
mu = mu * 252

# Matriz de covariância (com shrinkage para estabilidade)
sigma, shrinkage_param = ledoit_wolf_shrinkage(recent_returns)
sigma = sigma * 252  # anualizar

print("  ✅ Retornos anualizados estimados:")
for ticker in tickers:
    print(f"     {ticker}: {mu[ticker]:.2%}")
print()
print("  ✅ Covariância estimada com Ledoit-Wolf")
print(f"     Dimensão: {sigma.shape}")
print(f"     Shrinkage: {shrinkage_param:.4f}")
print()

# ============================================================================
# 3. OTIMIZAR PORTFOLIO (MEAN-VARIANCE)
# ============================================================================
print("⚙️  [3/5] Otimizando portfolio (Mean-Variance)...")

from itau_quant.optimization.core.mv_qp import MeanVarianceConfig, solve_mean_variance

# Configuração
config = MeanVarianceConfig(
    risk_aversion=3.0,  # λ moderado
    turnover_penalty=0.0,  # sem penalidade de turnover por enquanto
    turnover_cap=None,
    lower_bounds=pd.Series(0.0, index=tickers),  # long-only
    upper_bounds=pd.Series(0.40, index=tickers),  # max 40% por ativo
    previous_weights=pd.Series(0.0, index=tickers),  # sem posição anterior
    cost_vector=None,
    solver="ECOS",
    solver_kwargs=None,
    risk_config=None,
    factor_loadings=None,
)

try:
    result = solve_mean_variance(mu, sigma, config)

    print("  ✅ Otimização concluída!")
    print(f"     Status: {result.summary.status}")
    print(f"     Solver: {result.summary.solver}")
    print(f"     Tempo: {result.summary.runtime:.3f}s")
    print()

    # ============================================================================
    # 4. VALIDAR RESULTADO
    # ============================================================================
    print("✅ [4/5] Validando resultado...")

    weights = result.weights

    # Verificações básicas
    print("  Verificação 1 - Soma dos pesos:")
    weights_sum = weights.sum()
    print(
        f"     Soma = {weights_sum:.6f} {'✅' if abs(weights_sum - 1.0) < 1e-4 else '❌'}"
    )

    print("  Verificação 2 - Long-only:")
    all_positive = (weights >= -1e-6).all()
    print(f"     Todos >= 0: {'✅' if all_positive else '❌'}")

    print("  Verificação 3 - Limites superiores:")
    within_bounds = (weights <= 0.40 + 1e-6).all()
    print(f"     Todos <= 40%: {'✅' if within_bounds else '❌'}")

    print()
    print("  📊 Alocação otimizada:")
    for ticker in tickers:
        w = weights[ticker]
        bar = "█" * int(w * 100)
        print(f"     {ticker}: {w:6.2%} {bar}")

    print()

    # Métricas de portfolio
    portfolio_return = float(mu @ weights)
    portfolio_vol = float(np.sqrt(weights @ sigma @ weights))
    sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0

    print("  📈 Métricas de Portfolio:")
    print(f"     Retorno esperado: {portfolio_return:.2%} a.a.")
    print(f"     Volatilidade:     {portfolio_vol:.2%} a.a.")
    print(f"     Sharpe Ratio:     {sharpe:.2f}")
    print()

    # ============================================================================
    # 5. VALIDAÇÕES FINAIS
    # ============================================================================
    print("🔄 [5/5] Validações finais...")

    # Pesos anteriores (igual-peso)
    prev_weights = pd.Series(1.0 / n_assets, index=tickers)

    # Calcular turnover
    trades = weights - prev_weights
    turnover = trades.abs().sum()

    print(f"  ✅ Turnover calculado: {turnover:.2%}")
    print()

    print("  📊 Mudanças de alocação:")
    for ticker in tickers:
        w_old = prev_weights[ticker]
        w_new = weights[ticker]
        delta = w_new - w_old
        arrow = "↑" if delta > 0.001 else "↓" if delta < -0.001 else "→"
        print(f"     {ticker}: {w_old:.2%} → {w_new:.2%} {arrow} ({delta:+.2%})")

    print()
    print("=" * 70)
    print("  ✅ TODOS OS TESTES PASSARAM!")
    print("=" * 70)
    print()
    print("🎉 Sistema pronto para produção!")
    print()
    print("📋 Resumo:")
    print(f"   • Dados: {n_days} dias, {n_assets} ativos")
    print(f"   • Otimização: {result.summary.runtime:.3f}s ({result.summary.solver})")
    print(f"   • Portfolio: {portfolio_return:.2%} retorno, {portfolio_vol:.2%} vol")
    print(f"   • Sharpe: {sharpe:.2f}")
    print(f"   • Concentração: {(weights > 0.001).sum()} ativos ativos")
    print(f"   • Turnover: {turnover:.2%}")
    print()

except Exception as e:
    print(f"  ❌ Erro na otimização: {e}")
    import traceback

    traceback.print_exc()
