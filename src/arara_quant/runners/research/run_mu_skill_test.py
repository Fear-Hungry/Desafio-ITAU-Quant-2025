#!/usr/bin/env python
"""
PRISM-R - Teste de Skill do Estimador de μ

Valida se os estimadores de retorno esperado têm poder preditivo real
ou se estão apenas gerando ruído overfitado.

Decisão crítica: se IC < 0.05 e PSR < 60%, PARE de usar μ.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from arara_quant.config import get_settings
from arara_quant.data import get_arara_universe

SETTINGS = get_settings()

print("=" * 80)
print("  PRISM-R - Teste de Skill do Estimador μ")
print("  Pergunta: μ̂ prevê r_{t+1}?")
print("=" * 80)
print()

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

TICKERS = get_arara_universe() + ["BITO"]

START_DATE = datetime.now() - timedelta(days=5 * 365)
END_DATE = datetime.now()

WINDOW = 252  # 1 year
STEP = 21  # Monthly reestimation
N_TRIALS = 10  # Conservative estimate of strategies tested

# ============================================================================
# [1] CARREGAR DADOS
# ============================================================================

print("📥 [1/4] Carregando dados históricos...")
print(f"   Período: {START_DATE.date()} a {END_DATE.date()}")
print(f"   Tickers: {len(TICKERS)} ativos")

data = yf.download(
    TICKERS,
    start=START_DATE,
    end=END_DATE,
    progress=False,
    auto_adjust=True,
)

if "Close" in data.columns:
    prices = data["Close"]
elif isinstance(data.columns, pd.MultiIndex):
    prices = data.xs("Close", level=0, axis=1)
else:
    prices = data

if isinstance(prices, pd.Series):
    prices = prices.to_frame()

prices = prices.dropna(axis=1, how="all")
valid_tickers = list(prices.columns)

print(f"   ✅ Dados carregados: {len(prices)} dias, {len(valid_tickers)} ativos válidos")
print()

# ============================================================================
# [2] CALCULAR RETORNOS
# ============================================================================

print("📊 [2/4] Calculando retornos...")

returns = prices.pct_change().dropna()
returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")

print(f"   ✅ Retornos: {len(returns)} observações")
print()

# ============================================================================
# [3] DEFINIR ESTIMADORES A TESTAR
# ============================================================================

print("🔧 [3/4] Definindo estimadores de μ...")

from arara_quant.estimators.mu import huber_mean


def sample_mu_estimator(rets: pd.DataFrame) -> pd.Series:
    """Sample mean (baseline overfit)."""
    return rets.mean() * 252


def huber_mu_estimator(rets: pd.DataFrame) -> pd.Series:
    """Huber M-estimator (robust)."""
    try:
        return huber_mean(rets, delta=1.5) * 252
    except:
        return rets.mean() * 252


def zero_mu_estimator(rets: pd.DataFrame) -> pd.Series:
    """μ = 0 (null hypothesis)."""
    return pd.Series(0.0, index=rets.columns)


estimators = {
    "sample_mean": sample_mu_estimator,
    "huber_mean": huber_mu_estimator,
    "zero": zero_mu_estimator,
}

print(f"   ✅ {len(estimators)} estimadores definidos:")
for name in estimators:
    print(f"      • {name}")
print()

# ============================================================================
# [4] RODAR SKILL TEST
# ============================================================================

print("🧪 [4/4] Testando skill preditivo...")
print()

from arara_quant.diagnostics.mu_skill import skill_report

results = {}

for name, estimator in estimators.items():
    print(f"   🔬 Testando '{name}'...")

    try:
        report = skill_report(
            returns,
            estimator,
            window=WINDOW,
            step=STEP,
            n_trials=N_TRIALS,
            ic_threshold=0.05,
            psr_threshold=0.60,
        )

        results[name] = report

        print(f"      IC:          {report.ic_mean:+.4f} ± {report.ic_std:.4f} (p={report.ic_pval:.3f})")
        print(f"      IC Hit Rate: {report.ic_hit_rate:.1%}")
        print(f"      R²:          {report.r2:.4f} (adj: {report.r2_adj:.4f})")
        print(f"      β (μ→r):     {report.beta:.4f} (p={report.beta_pval:.3f})")
        print(f"      Sharpe:      {report.sharpe_forecast:.3f}")
        print(f"      PSR:         {report.psr:.2%}")
        print(f"      DSR:         {report.dsr:.2%}")
        print(f"      Skill?       {'✅ YES' if report.has_skill else '❌ NO'}")
        print()

    except Exception as e:
        print(f"      ❌ Erro: {e}")
        print()
        results[name] = None

print("=" * 80)
print("  📊 RESUMO E RECOMENDAÇÃO")
print("=" * 80)
print()

# Tabela comparativa
comparison = []
for name, report in results.items():
    if report is None:
        continue
    comparison.append({
        "Estimator": name,
        "IC": f"{report.ic_mean:+.4f}",
        "IC p-val": f"{report.ic_pval:.3f}",
        "R²": f"{report.r2:.4f}",
        "PSR": f"{report.psr:.2%}",
        "DSR": f"{report.dsr:.2%}",
        "Skill": "✅" if report.has_skill else "❌",
    })

df_comp = pd.DataFrame(comparison)
print(df_comp.to_string(index=False))
print()

# Decisão final
print("🎯 Decisão:")
print()

has_any_skill = any(r.has_skill for r in results.values() if r is not None)

if has_any_skill:
    best = max(
        [(name, r) for name, r in results.items() if r is not None and r.has_skill],
        key=lambda x: x[1].psr
    )
    print(f"   ✅ RECOMENDAÇÃO: Use '{best[0]}' (PSR={best[1].psr:.2%})")
    print(f"      {best[1].recommendation}")
else:
    print("   ⚠️  NENHUM ESTIMADOR TEM SKILL DETECTÁVEL!")
    print()
    print("   📋 Opções:")
    print("      1. Use μ=0 e otimize min-variance ou risk parity")
    print("      2. Shrink μ agressivamente (γ ≥ 0.90)")
    print("      3. Use Black-Litterman neutro (sem views)")
    print("      4. Invista em melhores features/dados")

print()

# Salvar resultados
results_dir = SETTINGS.results_dir
results_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = results_dir / f"mu_skill_test_{timestamp}.csv"

df_comp.to_csv(output_file, index=False)
print(f"💾 Resultados salvos: {output_file}")

print()
print("=" * 80)
print(f"  {'✅ SKILL TEST CONCLUÍDO!' if has_any_skill else '⚠️  SKILL TEST CONCLUÍDO - SEM SKILL'}")
print("=" * 80)
