import matplotlib

matplotlib.use("Agg")

import pandas as pd

from arara_quant.evaluation.plots import plot_walkforward_summary


def test_plot_walkforward_summary_builds_figure():
    split_metrics = pd.DataFrame(
        {
            "sharpe_ratio": [0.5, -0.2, 0.8],
            "turnover": [0.05, 0.08, 0.03],
            "max_drawdown": [-0.05, -0.12, -0.03],
            "cost_fraction": [0.0005, 0.0008, 0.0003],
            "total_return": [0.02, -0.01, 0.03],
        }
    )
    fig = plot_walkforward_summary(split_metrics)
    assert len(fig.axes) >= 4

