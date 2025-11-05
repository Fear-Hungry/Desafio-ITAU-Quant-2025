#!/usr/bin/env python
"""
Production Monitor - Sistema de Fallback Automático

Monitora métricas de risco e performa switch automático para 1/N
quando triggers são violados.

Triggers (operacionais, usando métricas diárias):
1. Sharpe 6M ≤ 0.0
2. CVaR 5% < -2% (daily) - valores mais negativos que -2% ativam fallback
   (equivalente a ~-32% anual; note que targets do projeto são anualizados: CVaR ≤ 8% a.a.)
3. Max DD < -10% - drawdowns piores que -10% ativam fallback

Nota: CVaR é mantido em escala diária para triggers operacionais (facilita monitoramento
intraday), mas reportado anualizado (CVaR_diário × √252) para comparação com targets.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class TriggerStatus:
    """Status dos triggers de fallback"""

    sharpe_6m_negative: bool
    cvar_breach: bool
    drawdown_breach: bool
    any_triggered: bool

    def to_dict(self) -> Dict[str, bool]:
        return {
            "sharpe_6m_negative": self.sharpe_6m_negative,
            "cvar_breach": self.cvar_breach,
            "drawdown_breach": self.drawdown_breach,
            "any_triggered": self.any_triggered,
        }


@dataclass
class PortfolioMetrics:
    """Métricas de portfolio para monitoramento"""

    sharpe_6m: float
    cvar_95: float  # CVaR 5% (negativo indica perda)
    max_dd: float  # Max drawdown (negativo)
    returns_6m: pd.Series


def calculate_portfolio_metrics(
    returns: pd.Series,
    lookback_days: int = 126,  # 6 meses ~ 126 dias úteis
) -> PortfolioMetrics:
    """
    Calcula métricas de portfolio para os últimos N dias.

    Parameters
    ----------
    returns : pd.Series
        Série temporal de retornos diários
    lookback_days : int
        Janela de lookback em dias úteis (default: 126 = 6M)

    Returns
    -------
    PortfolioMetrics
        Métricas calculadas
    """
    if len(returns) < lookback_days:
        lookback_days = len(returns)

    returns_6m = returns.tail(lookback_days)

    # Sharpe 6M (anualizado)
    mean_return = returns_6m.mean() * 252
    vol = returns_6m.std() * np.sqrt(252)
    sharpe_6m = mean_return / vol if vol > 0 else 0.0

    # CVaR 5% (daily) - média dos 5% piores retornos
    var_5_threshold = returns_6m.quantile(0.05)
    cvar_95 = returns_6m[returns_6m <= var_5_threshold].mean()

    # Max Drawdown
    cum_returns = (1 + returns_6m).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_dd = drawdown.min()

    return PortfolioMetrics(
        sharpe_6m=sharpe_6m,
        cvar_95=cvar_95,
        max_dd=max_dd,
        returns_6m=returns_6m,
    )


def evaluate_triggers(
    metrics: PortfolioMetrics,
    sharpe_threshold: float = 0.0,
    cvar_threshold: float = -0.02,  # -2% diário
    dd_threshold: float = -0.10,  # -10%
) -> TriggerStatus:
    """
    Avalia se algum trigger de fallback foi violado.

    Parameters
    ----------
    metrics : PortfolioMetrics
        Métricas calculadas do portfolio
    sharpe_threshold : float
        Limite inferior para Sharpe 6M (default: 0.0)
    cvar_threshold : float
        Limite inferior para CVaR 95% diário (default: -2%, equiv. ~-32% anual)
    dd_threshold : float
        Limite inferior para Max DD (default: -10%)

    Returns
    -------
    TriggerStatus
        Status de cada trigger
    """
    triggers = {
        "sharpe_6m_negative": metrics.sharpe_6m <= sharpe_threshold,
        "cvar_breach": metrics.cvar_95 < cvar_threshold,
        "drawdown_breach": metrics.max_dd < dd_threshold,
    }

    any_triggered = any(triggers.values())

    return TriggerStatus(**triggers, any_triggered=any_triggered)


def should_fallback_to_1N(
    returns: pd.Series,
    lookback_days: int = 126,
    sharpe_threshold: float = 0.0,
    cvar_threshold: float = -0.02,
    dd_threshold: float = -0.10,
    verbose: bool = True,
) -> tuple[bool, TriggerStatus, PortfolioMetrics]:
    """
    Decisão de fallback para 1/N baseada em triggers.

    Parameters
    ----------
    returns : pd.Series
        Série temporal de retornos do portfolio
    lookback_days : int
        Janela de lookback (default: 126 dias = 6M)
    sharpe_threshold : float
        Limite Sharpe 6M (default: 0.0)
    cvar_threshold : float
        Limite CVaR 95% (default: -2%)
    dd_threshold : float
        Limite Max DD (default: -10%)
    verbose : bool
        Imprimir diagnóstico

    Returns
    -------
    should_fallback : bool
        True se deve fazer fallback para 1/N
    trigger_status : TriggerStatus
        Status detalhado dos triggers
    metrics : PortfolioMetrics
        Métricas calculadas
    """
    # Calcular métricas
    metrics = calculate_portfolio_metrics(returns, lookback_days)

    # Avaliar triggers
    trigger_status = evaluate_triggers(
        metrics,
        sharpe_threshold=sharpe_threshold,
        cvar_threshold=cvar_threshold,
        dd_threshold=dd_threshold,
    )

    if verbose and trigger_status.any_triggered:
        print("⚠️  FALLBACK TRIGGER ATIVADO!")
        print(
            f"   Sharpe 6M: {metrics.sharpe_6m:.2f} (limite: {sharpe_threshold:.2f}) {'❌' if trigger_status.sharpe_6m_negative else '✅'}"
        )
        print(
            f"   CVaR 95%: {metrics.cvar_95:.2%} (limite: {cvar_threshold:.2%}) {'❌' if trigger_status.cvar_breach else '✅'}"
        )
        print(
            f"   Max DD: {metrics.max_dd:.2%} (limite: {dd_threshold:.2%}) {'❌' if trigger_status.drawdown_breach else '✅'}"
        )
        print()
        print("   → SWITCH PARA 1/N RECOMENDADO")
    elif verbose:
        print("✅ Todos os triggers OK - continuar com ERC")
        print(f"   Sharpe 6M: {metrics.sharpe_6m:.2f}")
        print(f"   CVaR 95%: {metrics.cvar_95:.2%}")
        print(f"   Max DD: {metrics.max_dd:.2%}")

    return trigger_status.any_triggered, trigger_status, metrics


# ============================================================================
# TESTES
# ============================================================================


def test_triggers():
    """Smoke test dos triggers"""
    print("=" * 80)
    print("  TESTE DE TRIGGERS DE FALLBACK")
    print("=" * 80)
    print()

    # Cenário 1: Portfolio saudável
    print("Cenário 1: Portfolio saudável")
    np.random.seed(42)
    healthy_returns = pd.Series(np.random.normal(0.0005, 0.01, 252))  # Sharpe ~0.5
    fallback, status, metrics = should_fallback_to_1N(healthy_returns, verbose=True)
    assert not fallback, "Portfolio saudável não deve fazer fallback!"
    print()

    # Cenário 2: Sharpe negativo
    print("Cenário 2: Sharpe negativo (retornos ruins)")
    bad_returns = pd.Series(np.random.normal(-0.001, 0.01, 252))  # Retorno negativo
    fallback, status, metrics = should_fallback_to_1N(bad_returns, verbose=True)
    assert fallback, "Sharpe negativo deve ativar fallback!"
    assert status.sharpe_6m_negative, "Trigger de Sharpe deve estar ativo"
    print()

    # Cenário 3: Drawdown severo
    print("Cenário 3: Drawdown severo")
    crash_returns = pd.Series([0.01] * 100 + [-0.05] * 30 + [0.01] * 122)  # Crash
    fallback, status, metrics = should_fallback_to_1N(crash_returns, verbose=True)
    assert fallback, "Drawdown severo deve ativar fallback!"
    assert status.drawdown_breach, "Trigger de DD deve estar ativo"
    print()

    # Cenário 4: CVaR alto (tail risk)
    print("Cenário 4: CVaR alto (tail risk)")
    tail_returns = pd.Series(
        np.concatenate(
            [
                np.random.normal(0.001, 0.005, 240),  # Dias normais
                np.random.normal(-0.03, 0.01, 12),  # Dias de crash (5%)
            ]
        )
    )
    fallback, status, metrics = should_fallback_to_1N(tail_returns, verbose=True)
    assert fallback, "Tail risk alto deve ativar fallback!"
    assert status.cvar_breach, "Trigger de CVaR deve estar ativo"
    print()

    print("=" * 80)
    print("✅ TODOS OS TESTES PASSARAM!")
    print("=" * 80)


if __name__ == "__main__":
    test_triggers()
