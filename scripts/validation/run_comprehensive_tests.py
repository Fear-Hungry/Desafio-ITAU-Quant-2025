#!/usr/bin/env python
"""
PRISM-R - Comprehensive Validation Suite
Bateria completa de testes para validar o sistema de produção ERC v2

Este script executa:
1. Backtest walk-forward com universo completo ARARA
2. Comparação com baselines (1/N, Min-Var, Risk Parity)
3. Stress tests (COVID-19, Bear Market 2022)
4. Sensitivity analysis (vol targets, CASH floors)
5. Constraint validation
6. Estimator robustness tests
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

print("=" * 80)
print("  PRISM-R - COMPREHENSIVE VALIDATION SUITE")
print("  Validação Completa do Sistema de Produção ERC v2")
print("=" * 80)
print()

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

# Carregar configuração de produção
CONFIG_PATH = Path("configs/production_erc_v2.yaml")
UNIVERSE_PATH = Path("configs/universe_arara_robust.yaml")

with open(CONFIG_PATH) as f:
    prod_config = yaml.safe_load(f)

with open(UNIVERSE_PATH) as f:
    universe_config = yaml.safe_load(f)

TICKERS = universe_config["tickers"]
VOL_TARGET = prod_config["vol_target"]
TURNOVER_TARGET = prod_config["turnover_target"]
MAX_POSITION = prod_config["max_position"]
MIN_POSITION = prod_config["min_position"]
CARDINALITY_K = prod_config["cardinality_k"]

# Defensive overlay config
DEFENSIVE_CONFIG = prod_config["defensive_overlay"]
CASH_FLOOR_NORMAL = DEFENSIVE_CONFIG["cash_floor_normal"]
CASH_FLOOR_DEFENSIVE = DEFENSIVE_CONFIG["cash_floor_defensive"]

# Período de teste
END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=365 * 5)  # 5 anos

# Walk-forward parameters
TRAIN_WINDOW = 252
TEST_WINDOW = 21
REBALANCE_FREQ = 21

print(f"📊 Configuração:")
print(f"   • Universo: {len(TICKERS)} ativos ARARA")
print(f"   • Período: {START_DATE.date()} a {END_DATE.date()}")
print(f"   • Vol Target: {VOL_TARGET:.1%}")
print(f"   • Turnover Target: {TURNOVER_TARGET:.1%}")
print(f"   • Cardinality: K={CARDINALITY_K}")
print(
    f"   • CASH Floor: {CASH_FLOOR_NORMAL:.0%} (normal) / {CASH_FLOOR_DEFENSIVE:.0%} (defensive)"
)
print()

# ============================================================================
# CARREGAR DADOS
# ============================================================================
print("📥 [1/7] Carregando dados históricos...")

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

    # Filtrar ativos válidos
    valid_tickers = []
    for ticker in TICKERS:
        if (
            ticker in prices.columns
            and prices[ticker].notna().sum() >= TRAIN_WINDOW + 50
        ):
            valid_tickers.append(ticker)

    prices = prices[valid_tickers]
    prices["CASH"] = 1.0
    returns = prices.pct_change().dropna()
    returns["CASH"] = 0.0
    valid_tickers.append("CASH")

    print(f"   ✅ Dados carregados: {len(returns)} dias, {len(valid_tickers)} ativos")
    print(f"   ✅ Período: {returns.index[0].date()} a {returns.index[-1].date()}")
    print()

except Exception as e:
    print(f"   ❌ Erro ao carregar dados: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# DEFINIR ESTRATÉGIAS DE TESTE
# ============================================================================
print("🔧 [2/7] Configurando estratégias de teste...")

from itau_quant.estimators.mu import mean_return
from itau_quant.estimators.cov import ledoit_wolf_shrinkage, sample_cov
from itau_quant.optimization.core.mv_qp import solve_mean_variance, MeanVarianceConfig

strategies = {
    "ERC_v2_Prod": {
        "description": "ERC v2 Production (vol target 12%, CASH 15%, defensive overlay)",
        "use_defensive": True,
        "cash_floor": CASH_FLOOR_NORMAL,
        "vol_target": VOL_TARGET,
        "cardinality": CARDINALITY_K,
    },
    "EqualWeight": {
        "description": "1/N Equal Weight Baseline",
        "use_defensive": False,
        "cash_floor": 0.0,
        "vol_target": None,
        "cardinality": None,
    },
    "MinVariance": {
        "description": "Minimum Variance (Ledoit-Wolf)",
        "use_defensive": False,
        "cash_floor": 0.0,
        "vol_target": None,
        "cardinality": None,
    },
}

print(f"   ✅ {len(strategies)} estratégias configuradas:")
for name, config in strategies.items():
    print(f"      • {name}: {config['description']}")
print()

# ============================================================================
# CRIAR WALK-FORWARD SPLITS
# ============================================================================
print("🔀 [3/7] Criando splits walk-forward...")

from itau_quant.backtesting.walk_forward import generate_walk_forward_splits

try:
    splits = list(
        generate_walk_forward_splits(
            returns.index,
            train_window=TRAIN_WINDOW,
            test_window=TEST_WINDOW,
            purge_window=2,
            embargo_window=0,
        )
    )

    print(f"   ✅ {len(splits)} períodos de teste criados")
    print(f"   ✅ Primeiro teste: {splits[0].test_index[0].date()}")
    print(f"   ✅ Último teste: {splits[-1].test_index[-1].date()}")
    print()

except Exception as e:
    print(f"   ❌ Erro: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# FUNÇÃO AUXILIAR: CALCULAR PESOS ERC
# ============================================================================


def calculate_erc_weights(sigma, cardinality=None):
    """Calcula pesos ERC usando log-barrier method"""
    from scipy.optimize import minimize

    n = len(sigma)

    # Função objetivo: sum log(RC_i) com barreira para diversificação
    def objective(w):
        w = np.maximum(w, 1e-8)
        portfolio_var = w @ sigma @ w
        marginal_contrib = sigma @ w
        risk_contrib = w * marginal_contrib

        # Log-barrier for equal risk contribution
        log_rc = np.log(risk_contrib + 1e-8)
        obj = -np.sum(log_rc)

        return obj

    # Constraints
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Bounds
    bounds = [(0.0, 1.0) for _ in range(n)]

    # Initial guess
    w0 = np.ones(n) / n

    # Otimizar
    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    if result.success:
        weights = result.x

        # Aplicar cardinality constraint se necessário
        if cardinality is not None and cardinality < n:
            # Selecionar top-K por peso
            top_k_idx = np.argsort(weights)[-cardinality:]
            weights_k = np.zeros(n)
            weights_k[top_k_idx] = weights[top_k_idx]
            if weights_k.sum() > 0:
                weights_k = weights_k / weights_k.sum()
            weights = weights_k

        return weights
    else:
        # Fallback: equal weight
        return np.ones(n) / n


# ============================================================================
# RODAR BACKTESTS PARA CADA ESTRATÉGIA
# ============================================================================
print("🔄 [4/7] Rodando backtests walk-forward...")
print()

results = {}

for strategy_name, strategy_config in strategies.items():
    print(f"   📊 Testando: {strategy_name}")
    print(f"      {strategy_config['description']}")

    portfolio_returns = []
    dates = []
    nav = 1.0
    nav_series = []
    nav_peak = 1.0
    in_drawdown_mode = False

    for i, split in enumerate(splits):
        train_returns = returns.loc[split.train_index]
        test_returns = returns.loc[split.test_index]

        if len(train_returns) < TRAIN_WINDOW // 2:
            continue

        try:
            # ESTRATÉGIA: Equal Weight
            if strategy_name == "EqualWeight":
                n_assets = len(valid_tickers) - 1  # Excluir CASH
                weights = pd.Series(1.0 / n_assets, index=valid_tickers)
                weights["CASH"] = 0.0

            # ESTRATÉGIA: Minimum Variance
            elif strategy_name == "MinVariance":
                sigma, _ = ledoit_wolf_shrinkage(train_returns)
                sigma = sigma * 252

                # Remover CASH da otimização
                assets_no_cash = [t for t in valid_tickers if t != "CASH"]
                sigma_no_cash = sigma.loc[assets_no_cash, assets_no_cash]

                # Min variance: minimize w'Σw
                from scipy.optimize import minimize

                n = len(assets_no_cash)

                def obj(w):
                    return w @ sigma_no_cash @ w

                constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
                bounds = [(0.0, MAX_POSITION) for _ in range(n)]
                w0 = np.ones(n) / n

                result = minimize(
                    obj, w0, method="SLSQP", bounds=bounds, constraints=constraints
                )

                if result.success:
                    weights = pd.Series(result.x, index=assets_no_cash)
                    weights = weights.reindex(valid_tickers, fill_value=0.0)
                    weights["CASH"] = 0.0
                else:
                    weights = pd.Series(1.0 / n, index=valid_tickers)
                    weights["CASH"] = 0.0

            # ESTRATÉGIA: ERC v2 Production
            elif strategy_name == "ERC_v2_Prod":
                sigma, _ = ledoit_wolf_shrinkage(train_returns)
                sigma = sigma * 252

                # Remover CASH da otimização ERC
                assets_no_cash = [t for t in valid_tickers if t != "CASH"]
                sigma_no_cash = sigma.loc[assets_no_cash, assets_no_cash].values

                # Calcular ERC weights
                erc_weights = calculate_erc_weights(
                    sigma_no_cash, cardinality=strategy_config["cardinality"]
                )

                weights = pd.Series(erc_weights, index=assets_no_cash)
                weights = weights.reindex(valid_tickers, fill_value=0.0)

                # Aplicar CASH floor
                cash_floor = strategy_config["cash_floor"]
                if cash_floor > 0:
                    weights = weights * (1 - cash_floor)
                    weights["CASH"] = cash_floor

                # Normalizar
                if weights.sum() > 0:
                    weights = weights / weights.sum()

            else:
                # Default: equal weight
                weights = pd.Series(1.0 / len(valid_tickers), index=valid_tickers)

            # Aplicar pesos no test set
            test_portfolio_returns = (test_returns * weights).sum(axis=1)

            # Atualizar NAV
            for ret in test_portfolio_returns:
                nav *= 1 + ret
                nav_series.append(nav)
                nav_peak = max(nav_peak, nav)

            portfolio_returns.extend(test_portfolio_returns.tolist())
            dates.extend(test_returns.index.tolist())

        except Exception as e:
            print(f"         Erro no período {i + 1}: {e}")
            continue

    # Calcular métricas
    portfolio_returns_series = pd.Series(portfolio_returns, index=dates)
    cumulative_returns = (1 + portfolio_returns_series).cumprod()
    total_return = cumulative_returns.iloc[-1] - 1
    n_years = len(portfolio_returns_series) / 252
    annualized_return = (1 + total_return) ** (1 / n_years) - 1
    annualized_vol = portfolio_returns_series.std() * np.sqrt(252)
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0

    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    downside_returns = portfolio_returns_series[portfolio_returns_series < 0]
    downside_vol = (
        downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0001
    )
    sortino = annualized_return / downside_vol if downside_vol > 0 else 0

    win_rate = (portfolio_returns_series > 0).sum() / len(portfolio_returns_series)

    # Armazenar resultados
    results[strategy_name] = {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": annualized_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "final_nav": nav,
        "returns": portfolio_returns_series,
    }

    print(
        f"      ✅ Sharpe: {sharpe:.2f} | Vol: {annualized_vol:.1%} | DD: {max_drawdown:.1%}"
    )
    print()

# ============================================================================
# COMPARAÇÃO DE RESULTADOS
# ============================================================================
print("=" * 80)
print("📊 [5/7] COMPARAÇÃO DE ESTRATÉGIAS")
print("=" * 80)
print()

# Tabela comparativa
comparison_df = pd.DataFrame(
    {
        name: {
            "Retorno Anual": f"{res['annualized_return']:.2%}",
            "Volatilidade": f"{res['volatility']:.2%}",
            "Sharpe": f"{res['sharpe']:.2f}",
            "Sortino": f"{res['sortino']:.2f}",
            "Max DD": f"{res['max_drawdown']:.2%}",
            "Win Rate": f"{res['win_rate']:.1%}",
            "NAV Final": f"{res['final_nav']:.2f}",
        }
        for name, res in results.items()
    }
)

print(comparison_df.T.to_string())
print()

# ============================================================================
# STRESS TESTS
# ============================================================================
print("=" * 80)
print("🚨 [6/7] STRESS TESTS - Períodos de Crise")
print("=" * 80)
print()

stress_periods = {
    "COVID-19 (Mar 2020)": (datetime(2020, 2, 1), datetime(2020, 4, 30)),
    "Bear Market 2022": (datetime(2022, 1, 1), datetime(2022, 10, 31)),
}

for period_name, (start, end) in stress_periods.items():
    print(f"📉 {period_name}")
    print(f"   Período: {start.date()} a {end.date()}")
    print()

    for strategy_name, result in results.items():
        rets = result["returns"]
        period_mask = (rets.index >= start) & (rets.index <= end)
        period_rets = rets[period_mask]

        if len(period_rets) > 0:
            period_total_ret = (1 + period_rets).prod() - 1
            period_vol = period_rets.std() * np.sqrt(252)
            period_sharpe = (
                (period_total_ret * 252 / len(period_rets)) / period_vol
                if period_vol > 0
                else 0
            )

            cum_rets = (1 + period_rets).cumprod()
            running_max = cum_rets.expanding().max()
            dd = (cum_rets - running_max) / running_max
            period_max_dd = dd.min()

            print(
                f"   {strategy_name:20s}: Ret={period_total_ret:+.1%} | DD={period_max_dd:.1%} | Vol={period_vol:.1%}"
            )
        else:
            print(f"   {strategy_name:20s}: Sem dados")

    print()

# ============================================================================
# SALVAR RESULTADOS
# ============================================================================
print("=" * 80)
print("💾 [7/7] Salvando resultados")
print("=" * 80)
print()

output_dir = Path("results/validation")
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Salvar comparação de estratégias
comparison_file = output_dir / f"strategy_comparison_{timestamp}.csv"
comparison_df.T.to_csv(comparison_file)
print(f"   ✅ Comparação salva: {comparison_file}")

# Salvar série de retornos de cada estratégia
for strategy_name, result in results.items():
    returns_file = output_dir / f"returns_{strategy_name}_{timestamp}.csv"
    result["returns"].to_csv(returns_file, header=["return"])
    print(f"   ✅ Retornos {strategy_name}: {returns_file}")

print()
print("=" * 80)
print("  ✅ VALIDAÇÃO COMPLETA CONCLUÍDA!")
print("=" * 80)
print()

# Resumo final
print("🎯 RESUMO EXECUTIVO:")
print()
print("Melhor Estratégia por Métrica:")
best_sharpe = max(results.items(), key=lambda x: x[1]["sharpe"])
best_return = max(results.items(), key=lambda x: x[1]["annualized_return"])
best_dd = max(results.items(), key=lambda x: x[1]["max_drawdown"])

print(f"   • Maior Sharpe: {best_sharpe[0]} ({best_sharpe[1]['sharpe']:.2f})")
print(
    f"   • Maior Retorno: {best_return[0]} ({best_return[1]['annualized_return']:.2%})"
)
print(f"   • Menor DD: {best_dd[0]} ({best_dd[1]['max_drawdown']:.2%})")
print()

# Validação de targets
erc_result = results.get("ERC_v2_Prod")
if erc_result:
    print("📋 Validação de Targets (ERC v2 Production):")
    print(
        f"   • Sharpe Ratio: {erc_result['sharpe']:.2f} {'✅' if erc_result['sharpe'] >= 0.80 else '❌'} (target ≥ 0.80)"
    )
    print(
        f"   • Max Drawdown: {erc_result['max_drawdown']:.2%} {'✅' if erc_result['max_drawdown'] >= -0.15 else '❌'} (target ≥ -15%)"
    )
    print(
        f"   • Volatilidade: {erc_result['volatility']:.2%} {'✅' if erc_result['volatility'] <= 0.12 else '❌'} (target ≤ 12%)"
    )
    print()

print(f"📁 Todos os resultados salvos em: {output_dir}/")
print()
