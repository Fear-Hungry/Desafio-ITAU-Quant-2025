"""Convenience exports for plotting utilities."""

from .diagnostics import (
    plot_drawdown_contributors,
    plot_parameter_sensitivity,
    plot_signal_distribution,
    plot_turnover_vs_cost,
    plot_weight_stability,
)
from .tearsheet import (
    TearsheetFigure,
    generate_tearsheet,
    plot_cumulative_returns,
    plot_drawdown,
    plot_risk_contribution,
    plot_rolling_sharpe,
    plot_rolling_volatility,
    plot_turnover,
)
from .walkforward import (
    plot_consistency_scatter,
    plot_parameter_evolution,
    plot_per_window_sharpe,
    plot_walkforward_summary,
)
from .oos import (
    BaselinePoint,
    build_default_baseline_points,
    plot_daily_returns_dashboard,
    plot_nav_cumulative,
    plot_risk_return_scatter,
    plot_underwater_drawdown,
)

__all__ = [
    "plot_cumulative_returns",
    "plot_drawdown",
    "plot_rolling_sharpe",
    "plot_rolling_volatility",
    "plot_turnover",
    "plot_risk_contribution",
    "generate_tearsheet",
    "TearsheetFigure",
    "plot_weight_stability",
    "plot_signal_distribution",
    "plot_parameter_sensitivity",
    "plot_turnover_vs_cost",
    "plot_drawdown_contributors",
    "plot_parameter_evolution",
    "plot_per_window_sharpe",
    "plot_consistency_scatter",
    "plot_walkforward_summary",
    "BaselinePoint",
    "build_default_baseline_points",
    "plot_nav_cumulative",
    "plot_underwater_drawdown",
    "plot_risk_return_scatter",
    "plot_daily_returns_dashboard",
]
