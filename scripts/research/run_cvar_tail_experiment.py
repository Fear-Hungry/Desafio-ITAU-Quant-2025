#!/usr/bin/env python
"""Compare mean-CVaR optimisation against Risk Parity and Equal-Weight.

This research script consumes the cached ARARA return matrix and performs a
walk-forward evaluation focused on tail control. Two CVaR variants are tested:
one with an explicit return target and another with a hard CVaR cap. Results
are written to ``results/cvar_experiment`` for inclusion in reports.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from arara_quant.evaluation.oos import (
    StrategySpec,
    compare_baselines,
    default_strategies,
)
from arara_quant.optimization.core.cvar_lp import CvarConfig, solve_cvar_lp

print("=" * 80)
print("  PRISM-R - Mean-CVaR vs Risk Parity Tail Experiment")
print("=" * 80)
print()

# ============================================================================
# CONFIGURAÇÃO DO EXPERIMENTO
# ============================================================================

DATA_PATH = Path("data/processed/returns_arara.parquet")

TRAIN_WINDOW = 252
TEST_WINDOW = 21
PURGE_WINDOW = 5
EMBARGO_WINDOW = 5

MAX_POSITION = 0.12
TRANSACTION_COST_BPS = 30

CVAR_ALPHA = 0.95
CVAR_TARGET_RETURN = 0.0004  # ~10% anualizado
CVAR_MAX = 0.06  # 6% tail loss (diário)
CVAR_RISK_AVERSION = 2.5
CVAR_RISK_AVERSION_LIMITED = 1.5
CVAR_TURNOVER_PENALTY = 0.02
CVAR_TURNOVER_CAP = 0.20
CVAR_SOLVER = "CLARABEL"

BOOTSTRAP_ITERATIONS = 0
CONFIDENCE = 0.95
BLOCK_SIZE = 21

print("📊 Parâmetros principais:")
print(f"   • Janela train/test: {TRAIN_WINDOW}/{TEST_WINDOW} dias")
print(f"   • Custos de transação: {TRANSACTION_COST_BPS} bps")
print(f"   • Max por ativo: {MAX_POSITION:.0%}")
print(f"   • CVaR α={CVAR_ALPHA:.2f}, limite {CVAR_MAX:.2%}")
print()

# ============================================================================
# 1. CARREGAR RETORNOS
# ============================================================================

if not DATA_PATH.exists():
    print(f"❌ Arquivo de retornos não encontrado: {DATA_PATH}")
    sys.exit(1)

returns = pd.read_parquet(DATA_PATH)
returns = returns.sort_index()

min_obs = TRAIN_WINDOW + 60
valid_cols = [c for c in returns.columns if returns[c].count() >= min_obs]
if not valid_cols:
    print("❌ Nenhum ativo possui observações suficientes para o backtest.")
    sys.exit(1)

returns = returns[valid_cols].fillna(0.0).astype(float)

start_date = returns.index.min().date()
end_date = returns.index.max().date()

print("📥 Dados carregados:")
print(f"   • Período: {start_date} a {end_date}")
print(f"   • Ativos elegíveis: {len(valid_cols)}")
print()

# ============================================================================
# 2. DEFINIR ESTRATÉGIAS
# ============================================================================

base_strategies = default_strategies(max_position=MAX_POSITION)

def _equal_weight(train_returns: pd.DataFrame, _: pd.Series | None) -> pd.Series:
    cols = train_returns.columns
    if len(cols) == 0:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(cols), index=cols, dtype=float)


def _prepare_window(data: pd.DataFrame) -> pd.DataFrame:
    """Limpa a janela de treino removendo NaNs generalizados."""

    cleaned = data.dropna(axis=0, how="any")
    if cleaned.empty:
        return data.fillna(0.0)
    return cleaned


def _build_cvar_strategy(*, use_target: bool) -> StrategySpec:
    label = "mean_cvar_target" if use_target else "mean_cvar_limit"

    def builder(train_returns: pd.DataFrame, prev_weights: pd.Series | None) -> pd.Series:
        window = _prepare_window(train_returns)
        if window.empty:
            return _equal_weight(train_returns, prev_weights)

        expected = window.mean()
        assets = expected.index

        prev = pd.Series(0.0, index=assets, dtype=float)
        if prev_weights is not None:
            prev = prev_weights.reindex(assets).fillna(0.0).astype(float)

        config = CvarConfig(
            alpha=CVAR_ALPHA,
            risk_aversion=CVAR_RISK_AVERSION if use_target else CVAR_RISK_AVERSION_LIMITED,
            long_only=True,
            lower_bounds=pd.Series(0.0, index=assets, dtype=float),
            upper_bounds=pd.Series(MAX_POSITION, index=assets, dtype=float),
            turnover_penalty=CVAR_TURNOVER_PENALTY,
            turnover_cap=CVAR_TURNOVER_CAP,
            previous_weights=prev,
            target_return=CVAR_TARGET_RETURN if use_target else None,
            max_cvar=CVAR_MAX,
            solver=CVAR_SOLVER,
            solver_kwargs={"max_iters": 10_000},
        )

        try:
            result = solve_cvar_lp(window, expected, config)
            weights = result.weights.reindex(assets).fillna(0.0)
        except Exception:
            return _equal_weight(train_returns, prev_weights)

        weights = weights.clip(lower=0.0, upper=MAX_POSITION)
        total = float(weights.sum())
        if total <= 0:
            return _equal_weight(train_returns, prev_weights)
        return weights / total

    return StrategySpec(label, builder)


strategies: list[StrategySpec] = [
    spec for spec in base_strategies if spec.name in {"equal_weight", "risk_parity"}
]
strategies.append(_build_cvar_strategy(use_target=True))
strategies.append(_build_cvar_strategy(use_target=False))

print("🔧 Estratégias avaliadas:")
for spec in strategies:
    print(f"   • {spec.name}")
print()

# ============================================================================
# 3. EXECUTAR WALK-FORWARD
# ============================================================================

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
    confidence=CONFIDENCE,
    block_size=BLOCK_SIZE,
    random_state=42,
)

metrics_df = oos_result.metrics.copy()
if "strategy" in metrics_df.columns:
    metrics_df = metrics_df.set_index("strategy")
elif metrics_df.index.name != "strategy":
    metrics_df = metrics_df.copy()
    metrics_df.index = metrics_df.index.astype(str)
    metrics_df.index.name = "strategy"

metrics = metrics_df.loc[[spec.name for spec in strategies]].copy()

if "risk_parity" in metrics.index:
    metrics["tail_improvement_vs_rp"] = (
        metrics.loc["risk_parity", "cvar_95"] - metrics["cvar_95"]
    )
else:
    metrics["tail_improvement_vs_rp"] = float("nan")

display_cols = ["sharpe", "annualized_return", "volatility", "cvar_95", "max_drawdown", "avg_turnover"]
print("=" * 80)
print("  📊 MÉTRICAS OUT-OF-SAMPLE (principal)")
print("=" * 80)
print(metrics[display_cols].to_string(float_format=lambda x: f"{x:6.4f}"))
print()

print("=" * 80)
print("  🪙 Ganho de cauda vs Risk Parity (Δ CVaR 95%)")
print("=" * 80)
print(metrics["tail_improvement_vs_rp"].to_string(float_format=lambda x: f"{x:6.4f}"))
print()

# ============================================================================
# 4. SALVAR RESULTADOS
# ============================================================================

output_dir = Path("results") / "cvar_experiment"
output_dir.mkdir(parents=True, exist_ok=True)

metrics_file = output_dir / "metrics_oos.csv"
metrics.reset_index().to_csv(metrics_file, index=False)

returns_file = output_dir / "strategy_returns_oos.parquet"
oos_result.returns.to_parquet(returns_file)

weights_dir = output_dir / "weights"
weights_dir.mkdir(exist_ok=True)
for strategy, snapshots in oos_result.weights.items():
    if not snapshots:
        continue
    df = pd.DataFrame({ts.strftime("%Y-%m-%d"): weights for ts, weights in snapshots})
    df.to_csv(weights_dir / f"{strategy}_weights.csv")

print("💾 Artefatos salvos em:")
print(f"   • {metrics_file}")
print(f"   • {returns_file}")
print(f"   • {weights_dir} (histórico de pesos)")
print()
print("Done ✅")
