#!/usr/bin/env python
"""
PRISM-R - Comparação de Estimadores de μ (Retorno Esperado)

Testa múltiplos estimadores de retorno esperado:
1. Sample mean (baseline overfit)
2. Huber mean (robust M-estimator)
3. Bayesian shrinkage to zero (conservador)
4. Black-Litterman neutro (sem views, apenas prior)

Objetivo: Escolher estimador que minimiza overfit mantendo power.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from itau_quant.data import get_arara_universe

print("=" * 80)
print("  PRISM-R - Comparação de Estimadores de μ")
print("  Teste Sistemático: Sample vs Huber vs Shrunk vs BL-Neutral")
print("=" * 80)
print()

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

TICKERS = get_arara_universe()

END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=365 * 3)
ESTIMATION_WINDOW = 252

# Parâmetros de otimização (FIXOS para comparação justa)
RISK_AVERSION = 4.0
MAX_POSITION = 0.10
TURNOVER_PENALTY = 0.0015
TURNOVER_CAP = 0.12
TRANSACTION_COST_BPS = 30

print("📊 Configuração:")
print(f"   • Universo: {len(TICKERS)} ativos")
print(f"   • Período: {START_DATE.date()} a {END_DATE.date()}")
print(f"   • Window: {ESTIMATION_WINDOW} dias")
print(f"   • Risk Aversion: {RISK_AVERSION} (fixo)")
print(f"   • Max Position: {MAX_POSITION:.0%} (fixo)")
print()

# ============================================================================
# 1. CARREGAR DADOS
# ============================================================================
print("📥 [1/4] Carregando dados...")

try:
    import yfinance as yf

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

    prices = prices.dropna(how="all").ffill().bfill()

    min_obs = ESTIMATION_WINDOW + 50
    valid_tickers = [
        t for t in TICKERS if t in prices.columns and prices[t].notna().sum() >= min_obs
    ]
    prices = prices[valid_tickers]

    returns = prices.pct_change().dropna()
    recent_returns = returns.tail(ESTIMATION_WINDOW)

    print(
        f"   ✅ Dados carregados: {len(prices)} dias, {len(valid_tickers)} ativos válidos"
    )
    print()

except Exception as e:
    print(f"   ❌ Erro: {e}")
    sys.exit(1)

# ============================================================================
# 2. ESTIMAR Σ (COMUM PARA TODOS)
# ============================================================================
print("📊 [2/4] Estimando Σ (Ledoit-Wolf, comum para todos)...")

from itau_quant.estimators.cov import ledoit_wolf_shrinkage

sigma, shrinkage = ledoit_wolf_shrinkage(recent_returns)
sigma_annual = sigma * 252

print(f"   ✅ Ledoit-Wolf shrinkage: {shrinkage:.4f}")
print()

# ============================================================================
# 3. ESTIMAR μ COM MÚLTIPLOS MÉTODOS
# ============================================================================
print("📈 [3/4] Estimando μ com 4 métodos diferentes...")

from itau_quant.estimators.bl import black_litterman, reverse_optimization
from itau_quant.estimators.mu import bayesian_shrinkage_mean, huber_mean, mean_return

# Método 1: Sample mean (baseline)
print("   [1/4] Sample mean (baseline overfit)...")
mu_sample = mean_return(recent_returns, method="simple") * 252
print(f"         Média: {mu_sample.mean():.2%}, Std: {mu_sample.std():.2%}")

# Método 2: Huber mean (robust)
print("   [2/4] Huber mean (robust M-estimator, delta=1.5)...")
mu_huber_daily, weights_eff = huber_mean(recent_returns, c=1.5)
mu_huber = mu_huber_daily * 252
outliers_downweighted = (weights_eff < 0.5).sum().sum()
print(f"         Média: {mu_huber.mean():.2%}, Std: {mu_huber.std():.2%}")
print(f"         Outliers down-weighted: {outliers_downweighted} observações")

# Método 3: Bayesian shrinkage to zero
print("   [3/4] Bayesian shrinkage to zero (strength=0.5)...")
mu_shrunk_daily = bayesian_shrinkage_mean(recent_returns, prior=0.0, strength=0.5)
mu_shrunk = mu_shrunk_daily * 252
print(f"         Média: {mu_shrunk.mean():.2%}, Std: {mu_shrunk.std():.2%}")

# Método 4: Black-Litterman neutro (sem views)
print("   [4/4] Black-Litterman neutro (sem views, tau=0.025)...")
market_weights = pd.Series(1 / len(valid_tickers), index=valid_tickers)
pi_prior_daily, delta = reverse_optimization(
    weights=market_weights,
    cov=sigma,
    risk_aversion=RISK_AVERSION,
)
pi_prior = pi_prior_daily * 252

bl_result = black_litterman(
    cov=sigma_annual,
    pi=pi_prior,
    views=[],
    tau=0.025,
    add_mean_uncertainty=True,
)
mu_bl = bl_result["mu_bl"]
print(f"         Média: {mu_bl.mean():.2%}, Std: {mu_bl.std():.2%}")

print()

# ============================================================================
# 4. OTIMIZAR PORTFOLIO COM CADA ESTIMADOR
# ============================================================================
print("⚙️  [4/4] Otimizando portfolio com cada estimador...")

from itau_quant.optimization.core.mv_qp import MeanVarianceConfig, solve_mean_variance

ESTIMATORS = {
    "sample": mu_sample,
    "huber": mu_huber,
    "shrunk_50": mu_shrunk,
    "bl_neutral": mu_bl,
}

cost_vector = pd.Series(TRANSACTION_COST_BPS / 10000, index=valid_tickers)

results = {}
for name, mu in ESTIMATORS.items():
    print(f"\n   🔧 [{name}] Otimizando...")

    config = MeanVarianceConfig(
        risk_aversion=RISK_AVERSION,
        turnover_penalty=TURNOVER_PENALTY,
        turnover_cap=None,  # Defina (ex.: 0.10) se quiser capear o turnover
        lower_bounds=pd.Series(0.0, index=valid_tickers),
        upper_bounds=pd.Series(MAX_POSITION, index=valid_tickers),
        previous_weights=pd.Series(0.0, index=valid_tickers),
        cost_vector=cost_vector,
        solver="CLARABEL",
    )

    try:
        result = solve_mean_variance(mu, sigma_annual, config)

        if result.summary.is_optimal():
            weights = result.weights

            # Métricas
            port_ret = float(mu @ weights)
            port_vol = float(np.sqrt(weights @ sigma_annual @ weights))
            sharpe = port_ret / port_vol if port_vol > 0 else 0

            n_active = (weights > 0.01).sum()
            herfindahl = (weights**2).sum()
            effective_n = 1.0 / herfindahl if herfindahl > 0 else 0

            # Número de ativos no teto
            at_ceiling = (weights >= MAX_POSITION * 0.99).sum()

            results[name] = {
                "weights": weights,
                "return": port_ret,
                "vol": port_vol,
                "sharpe": sharpe,
                "n_active": n_active,
                "effective_n": effective_n,
                "at_ceiling": at_ceiling,
                "solver_time": result.summary.runtime,
            }

            print(f"       Status: {result.summary.status}")
            print(
                f"       Sharpe: {sharpe:.2f}, Vol: {port_vol:.1%}, Ret: {port_ret:.1%}"
            )
            print(
                f"       N_active: {n_active}, N_eff: {effective_n:.1f}, At ceiling: {at_ceiling}"
            )
        else:
            print(f"       ⚠️  Status: {result.summary.status} (não optimal)")
            results[name] = None

    except Exception as e:
        print(f"       ❌ Erro: {e}")
        results[name] = None

print()

# ============================================================================
# ANALISAR E COMPARAR
# ============================================================================
print("=" * 80)
print("  📊 COMPARAÇÃO DE RESULTADOS")
print("=" * 80)
print()

comparison_rows = []
for name, res in results.items():
    if res is not None:
        comparison_rows.append(
            {
                "Estimator": name,
                "Return (ann)": f"{res['return']:.2%}",
                "Vol (ann)": f"{res['vol']:.2%}",
                "Sharpe": f"{res['sharpe']:.2f}",
                "N_active": res["n_active"],
                "N_eff": f"{res['effective_n']:.1f}",
                "At Ceiling": res["at_ceiling"],
                "Time (s)": f"{res['solver_time']:.3f}",
            }
        )

comparison_df = pd.DataFrame(comparison_rows)
print(comparison_df.to_string(index=False))
print()

# Identificar melhor estimador
valid_results = {k: v for k, v in results.items() if v is not None}
if valid_results:
    best_sharpe_name = max(valid_results, key=lambda k: valid_results[k]["sharpe"])
    best_neff_name = max(valid_results, key=lambda k: valid_results[k]["effective_n"])
    least_ceiling_name = min(
        valid_results, key=lambda k: valid_results[k]["at_ceiling"]
    )

    print("🏆 Rankings:")
    print(
        f"   • Melhor Sharpe: {best_sharpe_name} ({valid_results[best_sharpe_name]['sharpe']:.2f})"
    )
    print(
        f"   • Melhor diversificação (N_eff): {best_neff_name} ({valid_results[best_neff_name]['effective_n']:.1f})"
    )
    print(
        f"   • Menos cap-banging: {least_ceiling_name} ({valid_results[least_ceiling_name]['at_ceiling']} ativos)"
    )
    print()

    print("💡 Recomendação:")
    if valid_results["huber"]["at_ceiling"] < valid_results["sample"]["at_ceiling"]:
        print(
            f"   ✅ Use HUBER: menos cap-banging ({valid_results['huber']['at_ceiling']} vs {valid_results['sample']['at_ceiling']})"
        )
    if valid_results["sample"]["sharpe"] > 2.0:
        print("   ⚠️  Sample mean com Sharpe > 2.0 → provável overfit!")
    if valid_results.get("shrunk_50") and valid_results["shrunk_50"]["sharpe"] > 0.8:
        print(
            f"   ✅ Shrunk_50 com Sharpe {valid_results['shrunk_50']['sharpe']:.2f} → conservador mas realista"
        )
    print()

# ============================================================================
# SALVAR RESULTADOS
# ============================================================================
print("💾 Salvando resultados...")

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Salvar comparação
comparison_file = output_dir / f"estimator_comparison_{timestamp}.csv"
comparison_df.to_csv(comparison_file, index=False)
print(f"   ✅ Comparação salva: {comparison_file}")

# Salvar pesos de cada estimador
for name, res in results.items():
    if res is not None:
        weights_df = pd.DataFrame(
            {
                "ticker": res["weights"].index,
                "weight": res["weights"].values,
            }
        ).sort_values("weight", ascending=False)

        weights_file = output_dir / f"weights_{name}_{timestamp}.csv"
        weights_df.to_csv(weights_file, index=False)
        print(f"   ✅ Pesos ({name}) salvos: {weights_file}")

print()
print("=" * 80)
print("  ✅ COMPARAÇÃO DE ESTIMADORES CONCLUÍDA!")
print("=" * 80)
print()
print("🎯 Próximo passo:")
print("   • Escolha o estimador baseado em:")
print("     1. Sharpe ex-ante razoável (< 2.0)")
print("     2. Baixo cap-banging (at_ceiling < 3)")
print("     3. Alta diversificação (N_eff ≥ 10)")
print("   • Rode walk-forward backtest com o estimador escolhido")
print("   • Valide que Sharpe OOS ≥ baseline + 0.2")
print()
