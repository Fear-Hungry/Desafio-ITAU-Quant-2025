#!/usr/bin/env python
"""
PRISM-R - Estimator Robustness Tests
Testes de robustez dos estimadores de covariância

Este script compara:
1. Sample covariance vs Ledoit-Wolf shrinkage
2. Diferentes métodos de shrinkage
3. Estabilidade numérica (condition number)
4. Impacto nos pesos otimizados
"""

import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yaml

from arara_quant.config import get_settings

SETTINGS = get_settings()

print("=" * 80)
print("  PRISM-R - ESTIMATOR ROBUSTNESS TESTS")
print("  Validação de Estimadores de Covariância")
print("=" * 80)
print()

# ============================================================================
# CARREGAR DADOS
# ============================================================================
print("📥 [1/4] Carregando dados...")

UNIVERSE_PATH = SETTINGS.configs_dir / "universe" / "universe_arara_robust.yaml"

with open(UNIVERSE_PATH) as f:
    universe_config = yaml.safe_load(f)

TICKERS = universe_config["tickers"]

END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=365 * 3)

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

    valid_tickers = []
    for ticker in TICKERS:
        if ticker in prices.columns and prices[ticker].notna().sum() >= 252:
            valid_tickers.append(ticker)

    prices = prices[valid_tickers]
    returns = prices.pct_change().dropna()

    # Usar janela de 1 ano para testes
    returns_window = returns.iloc[-252:]

    print(
        f"   ✅ Dados carregados: {len(returns_window)} dias, {len(valid_tickers)} ativos"
    )
    print()

except Exception as e:
    print(f"   ❌ Erro: {e}")
    sys.exit(1)

# ============================================================================
# TESTE 1: SAMPLE COV VS LEDOIT-WOLF
# ============================================================================
print("=" * 80)
print("🔍 [2/4] TESTE 1: Sample Cov vs Ledoit-Wolf")
print("=" * 80)
print()

from arara_quant.estimators.cov import ledoit_wolf_shrinkage, sample_cov

# Sample covariance
sample_sigma = sample_cov(returns_window, ddof=1) * 252

# Ledoit-Wolf shrinkage
lw_sigma, lw_shrinkage = ledoit_wolf_shrinkage(returns_window)
lw_sigma = lw_sigma * 252

print("   Sample Covariance:")
sample_eigenvalues = np.linalg.eigvalsh(sample_sigma.values)
sample_cond = np.linalg.cond(sample_sigma.values)
print(f"      • Condition number: {sample_cond:.2e}")
print(f"      • Min eigenvalue: {sample_eigenvalues.min():.6f}")
print(f"      • Max eigenvalue: {sample_eigenvalues.max():.6f}")
print(f"      • Negative eigenvalues: {(sample_eigenvalues < 0).sum()}")
print()

print("   Ledoit-Wolf Shrinkage:")
lw_eigenvalues = np.linalg.eigvalsh(lw_sigma.values)
lw_cond = np.linalg.cond(lw_sigma.values)
print(f"      • Shrinkage intensity: {lw_shrinkage:.4f}")
print(f"      • Condition number: {lw_cond:.2e}")
print(f"      • Min eigenvalue: {lw_eigenvalues.min():.6f}")
print(f"      • Max eigenvalue: {lw_eigenvalues.max():.6f}")
print(f"      • Negative eigenvalues: {(lw_eigenvalues < 0).sum()}")
print()

# Comparação
improvement = (sample_cond - lw_cond) / sample_cond * 100
print(f"   Melhoria no condition number: {improvement:.1f}%")
print()

if lw_cond < 1e10:
    print("   ✅ PASSED: Ledoit-Wolf produz matriz bem-condicionada")
else:
    print("   ⚠️  WARNING: Matriz ainda mal-condicionada")

print()

# ============================================================================
# TESTE 2: IMPACTO NOS PESOS OTIMIZADOS
# ============================================================================
print("=" * 80)
print("🔍 [3/4] TESTE 2: Impacto nos Pesos Otimizados")
print("=" * 80)
print()

from scipy.optimize import minimize


def calculate_min_var_weights(sigma):
    """Calcula pesos de mínima variância"""
    n = len(sigma)

    def objective(w):
        return w @ sigma @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0) for _ in range(n)]
    w0 = np.ones(n) / n

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000},
    )

    if result.success:
        return result.x
    else:
        return np.ones(n) / n


# Calcular pesos com sample cov
try:
    weights_sample = calculate_min_var_weights(sample_sigma.values)
    weights_sample_series = pd.Series(weights_sample, index=sample_sigma.index)

    # Métricas
    n_active_sample = (weights_sample > 1e-4).sum()
    herfindahl_sample = (weights_sample**2).sum()
    n_eff_sample = 1 / herfindahl_sample
    max_weight_sample = weights_sample.max()

    print("   Pesos com Sample Cov:")
    print(f"      • N ativos ativos: {n_active_sample}")
    print(f"      • N effective: {n_eff_sample:.1f}")
    print(f"      • Max weight: {max_weight_sample:.1%}")
    print()

except Exception as e:
    print(f"   ❌ Sample Cov falhou: {e}")
    weights_sample_series = None
    print()

# Calcular pesos com Ledoit-Wolf
try:
    weights_lw = calculate_min_var_weights(lw_sigma.values)
    weights_lw_series = pd.Series(weights_lw, index=lw_sigma.index)

    # Métricas
    n_active_lw = (weights_lw > 1e-4).sum()
    herfindahl_lw = (weights_lw**2).sum()
    n_eff_lw = 1 / herfindahl_lw
    max_weight_lw = weights_lw.max()

    print("   Pesos com Ledoit-Wolf:")
    print(f"      • N ativos ativos: {n_active_lw}")
    print(f"      • N effective: {n_eff_lw:.1f}")
    print(f"      • Max weight: {max_weight_lw:.1%}")
    print()

except Exception as e:
    print(f"   ❌ Ledoit-Wolf falhou: {e}")
    weights_lw_series = None
    print()

# Comparar pesos
if weights_sample_series is not None and weights_lw_series is not None:
    weight_diff = np.abs(weights_sample - weights_lw).sum()
    correlation = np.corrcoef(weights_sample, weights_lw)[0, 1]

    print("   Comparação de Pesos:")
    print(f"      • L1 difference: {weight_diff:.2f}")
    print(f"      • Correlation: {correlation:.3f}")
    print()

    if correlation > 0.7:
        print("   ✅ PASSED: Pesos correlacionados (estável)")
    else:
        print("   ⚠️  WARNING: Pesos pouco correlacionados (instável)")

print()

# ============================================================================
# TESTE 3: ESTABILIDADE TEMPORAL
# ============================================================================
print("=" * 80)
print("🔍 [4/4] TESTE 3: Estabilidade Temporal")
print("=" * 80)
print()

# Calcular covariância em janelas sobrepostas
windows = [
    ("Últimos 1 ano", returns.iloc[-252:]),
    ("Últimos 6 meses", returns.iloc[-126:]),
    ("Últimos 3 meses", returns.iloc[-63:]),
]

cond_numbers = []

for window_name, window_data in windows:
    if len(window_data) < 50:
        continue

    sigma, shrinkage = ledoit_wolf_shrinkage(window_data)
    cond = np.linalg.cond(sigma.values)
    cond_numbers.append(cond)

    print(f"   {window_name}:")
    print(f"      • Shrinkage: {shrinkage:.4f}")
    print(f"      • Condition number: {cond:.2e}")
    print()

# Estabilidade
if len(cond_numbers) > 1:
    cond_std = np.std(cond_numbers)
    cond_mean = np.mean(cond_numbers)
    cv = cond_std / cond_mean if cond_mean > 0 else 0

    print("   Estabilidade:")
    print(f"      • CV(condition number): {cv:.2f}")
    print()

    if cv < 0.5:
        print("   ✅ PASSED: Estimador estável ao longo do tempo")
    else:
        print("   ⚠️  WARNING: Estimador instável")

print()

# ============================================================================
# TESTE 4: POSITIVE DEFINITENESS
# ============================================================================
print("=" * 80)
print("🔍 TESTE 4: Positive Definiteness")
print("=" * 80)
print()


def is_positive_definite(matrix):
    """Verifica se matriz é positive definite"""
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


sample_pd = is_positive_definite(sample_sigma.values)
lw_pd = is_positive_definite(lw_sigma.values)

print(f"   Sample Cov PD: {'✅ YES' if sample_pd else '❌ NO'}")
print(f"   Ledoit-Wolf PD: {'✅ YES' if lw_pd else '❌ NO'}")
print()

if lw_pd:
    print("   ✅ PASSED: Ledoit-Wolf garante PD")
else:
    print("   ❌ FAILED: Ledoit-Wolf não é PD")

print()

# ============================================================================
# RESUMO FINAL
# ============================================================================
print("=" * 80)
print("  📋 RESUMO DE VALIDAÇÃO")
print("=" * 80)
print()

all_tests = {
    "Condition Number < 1e10": lw_cond < 1e10,
    "Positive Definite": lw_pd,
    "Pesos Estáveis": correlation > 0.7
    if weights_sample_series is not None and weights_lw_series is not None
    else False,
    "Estimador Temporalmente Estável": cv < 0.5 if len(cond_numbers) > 1 else True,
}

n_passed = sum(all_tests.values())
n_total = len(all_tests)

print(f"   Testes Passados: {n_passed}/{n_total}")
print()

for test_name, passed in all_tests.items():
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"      {status}: {test_name}")

print()

if n_passed == n_total:
    print("   🎉 TODOS OS TESTES DE ESTIMADORES PASSARAM!")
    print()
    print("   💡 RECOMENDAÇÃO: Continue usando Ledoit-Wolf shrinkage")
else:
    print(f"   ⚠️  {n_total - n_passed} TESTE(S) FALHARAM")
    print()
    print("   💡 RECOMENDAÇÃO: Investigar estimadores alternativos")

print()
print("=" * 80)
