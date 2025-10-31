#!/usr/bin/env python
"""Walk-forward OOS comparison against baseline strategies.

This script runs a multi-strategy walk-forward evaluation using the helper
utilities in ``itau_quant.evaluation.oos``. It mirrors the PRD requirement of
comparing the robust portfolio against classical baselines (1/N, min-var, ERC,
60/40, HRP) and records stress-period diagnostics for COVID-19 and 2022
inflation shocks.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

print("=" * 80)
print("  PRISM-R - Comparação de Estratégias Baseline (OOS)")
print("  Walk-Forward Backtest com Múltiplas Estratégias")
print("=" * 80)
print()

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

TICKERS = [
    "SPY",
    "QQQ",
    "IWM",
    "VTV",
    "VUG",
    "EFA",
    "VGK",
    "EWJ",
    "EWU",
    "EWG",
    "EEM",
    "VWO",
    "EWZ",
    "FXI",
    "INDA",
    "TLT",
    "IEF",
    "SHY",
    "LQD",
    "HYG",
    "EMB",
    "GLD",
    "SLV",
    "DBC",
    "USO",
    "VNQ",
    "VNQI",
    "IBIT",
    "ETHA",
]

END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=365 * 5)

TRAIN_WINDOW = 252
TEST_WINDOW = 21
PURGE_WINDOW = 5
EMBARGO_WINDOW = 5

MAX_POSITION = 0.10
TRANSACTION_COST_BPS = 30
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_BLOCK = 21
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_SEED = 1234

print("📊 Configuração:")
print(f"   • Universo: {len(TICKERS)} ativos")
print(f"   • Período: {START_DATE.date()} a {END_DATE.date()}")
print(f"   • Train window: {TRAIN_WINDOW} dias")
print(f"   • Test window: {TEST_WINDOW} dias")
print(f"   • Purge/Embargo: {PURGE_WINDOW}/{EMBARGO_WINDOW} dias")
print(f"   • Transaction costs: {TRANSACTION_COST_BPS} bps")
print()

# ============================================================================
# 1. CARREGAR DADOS
# ============================================================================
print("📥 [1/3] Carregando dados...")

try:
    import yfinance as yf

    data = yf.download(
        tickers=TICKERS,
        start=START_DATE - timedelta(days=400),
        end=END_DATE,
        progress=False,
        auto_adjust=True,
    )

    prices = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    prices = prices.dropna(how="all").ffill().bfill()

    min_obs = TRAIN_WINDOW + 50
    valid_tickers = [t for t in TICKERS if prices[t].notna().sum() >= min_obs]
    prices = prices[valid_tickers]
    returns = prices.pct_change().dropna()

    print(f"   ✅ Dados carregados: {len(prices)} dias, {len(valid_tickers)} ativos")
    print(f"   ✅ Período efetivo: {returns.index[0].date()} a {returns.index[-1].date()}")
    print()

except Exception as exc:  # pragma: no cover - network dependency
    print(f"   ❌ Erro ao carregar dados: {exc}")
    sys.exit(1)

# ============================================================================
# 2. DEFINIR ESTRATÉGIAS E RODAR WALK-FORWARD
# ============================================================================
print("🔧 [2/3] Preparando avaliação OOS...")

from itau_quant.evaluation.oos import compare_baselines, default_strategies, stress_test

strategies = default_strategies(max_position=MAX_POSITION, shrink_strength=0.5)
print(f"   ✅ {len(strategies)} estratégias definidas:")
for spec in strategies:
    print(f"      • {spec.name}")
print()

oos_result = compare_baselines(
    returns,
    strategies=strategies,
    train_window=TRAIN_WINDOW,
    test_window=TEST_WINDOW,
    purge_window=PURGE_WINDOW,
    embargo_window=EMBARGO_WINDOW,
    costs_bps=TRANSACTION_COST_BPS,
    max_position=MAX_POSITION,
    bootstrap_iterations=BOOTSTRAP_ITERATIONS,
    confidence=BOOTSTRAP_CONFIDENCE,
    block_size=BOOTSTRAP_BLOCK,
    random_state=BOOTSTRAP_SEED,
)

metrics_df = oos_result.metrics.sort_values("sharpe", ascending=False)

print("=" * 80)
print("  📊 MÉTRICAS OUT-OF-SAMPLE")
print("=" * 80)
print(metrics_df.to_string(float_format=lambda x: f"{x:.4f}"))
print()

# ============================================================================
# 3. TESTES DE ESTRESSE
# ============================================================================
stress_periods = {
    "covid_crash": ("2020-02-19", "2020-04-30"),
    "inflation_2022": ("2022-01-01", "2022-12-31"),
    "banking_stress_2023": ("2023-03-01", "2023-05-31"),
}

stress_df = stress_test(oos_result.returns, stress_periods)
if not stress_df.empty:
    print("=" * 80)
    print("  🧪 TESTES DE ESTRESSE")
    print("=" * 80)
    print(stress_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print()
else:
    print("⚠️  Nenhum período de estresse coincidiu com a janela de avaliação.")
    print()

# Turnover logs por rebalance
turnover_records: list[dict[str, object]] = []
for name, weight_snapshots in oos_result.weights.items():
    series_turnovers = oos_result.turnovers.get(name, [])
    for (date, _weights), turnover in zip(weight_snapshots, series_turnovers):
        turnover_records.append({
            "date": pd.Timestamp(date),
            "strategy": name,
            "turnover": float(turnover),
        })

turnover_df = pd.DataFrame(turnover_records)
if not turnover_df.empty:
    turnover_pivot = turnover_df.pivot(index="date", columns="strategy", values="turnover").sort_index()
else:
    turnover_pivot = pd.DataFrame()

# ============================================================================
# SALVAR RESULTADOS
# ============================================================================
output_dir = Path("results") / "baselines"
output_dir.mkdir(parents=True, exist_ok=True)

returns_file = output_dir / "baseline_returns_oos.parquet"
oos_result.returns.to_parquet(returns_file)
print(f"   💾 Retornos OOS salvos em: {returns_file}")

metrics_file = output_dir / "baseline_metrics_oos.csv"
metrics_df.to_csv(metrics_file)
print(f"   💾 Métricas salvas em: {metrics_file}")

if not stress_df.empty:
    stress_file = output_dir / "baseline_stress_tests.csv"
    stress_df.to_csv(stress_file, index=False)
    print(f"   💾 Stress tests salvos em: {stress_file}")

if not turnover_pivot.empty:
    turnover_file = output_dir / "baseline_turnover_oos.csv"
    turnover_pivot.to_csv(turnover_file)
    print(f"   💾 Turnovers salvos em: {turnover_file}")

print()
print("Done ✅")
