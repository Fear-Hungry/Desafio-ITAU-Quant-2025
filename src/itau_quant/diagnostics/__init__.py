"""Diagnostic tools for strategy validation."""

from .mu_skill import (
    information_coefficient,
    predictive_r2,
    probabilistic_sharpe_ratio,
    skill_report,
)

__all__ = [
    "information_coefficient",
    "predictive_r2",
    "probabilistic_sharpe_ratio",
    "skill_report",
]
