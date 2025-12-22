"""Legacy risk parity entry points (backwards-compatible wrapper).

The canonical implementation lives in ``arara_quant.optimization.risk_parity``.
"""

from __future__ import annotations

from arara_quant.optimization.risk_parity import (
    RiskParityResult,
    cluster_risk_parity,
    iterative_risk_parity,
    risk_contribution,
    risk_parity,
    solve_log_barrier,
)

__all__ = [
    "RiskParityResult",
    "risk_contribution",
    "solve_log_barrier",
    "iterative_risk_parity",
    "cluster_risk_parity",
    "risk_parity",
]
