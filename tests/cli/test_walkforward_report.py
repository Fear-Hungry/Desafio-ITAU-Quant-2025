from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from itau_quant.backtesting.engine import BacktestResult
from itau_quant.cli import _generate_wf_report


def _build_result(tmp_path: Path) -> BacktestResult:
    dates = pd.date_range("2020-01-01", periods=2, freq="ME")
    weights = pd.DataFrame(
        {"AAA": [0.6, 0.55], "BBB": [0.4, 0.45]},
        index=pd.Index(dates, name="date"),
    )
    trades = pd.DataFrame(
        {
            "date": dates,
            "turnover": [0.05, 0.02],
            "cost": [0.0003, 0.0001],
        }
    )
    split_metrics = pd.DataFrame(
        {
            "train_start": ["2019-01-01", "2019-02-01"],
            "train_end": ["2019-12-31", "2020-01-31"],
            "test_start": ["2020-01-01", "2020-02-01"],
            "test_end": ["2020-01-31", "2020-02-28"],
            "total_return": [0.02, 0.01],
            "annualized_return": [0.24, 0.12],
            "annualized_volatility": [0.10, 0.08],
            "sharpe_ratio": [1.2, 0.9],
            "max_drawdown": [-0.05, -0.04],
            "cumulative_nav": [1.02, 1.03],
            "turnover": [0.05, 0.02],
            "cost_fraction": [0.0003, 0.0001],
        }
    )

    return BacktestResult(
        config_path=tmp_path / "dummy.yaml",
        environment="test",
        base_currency="USD",
        dry_run=False,
        weights=weights,
        trades=trades,
        split_metrics=split_metrics,
    )


def test_generate_wf_report_exports_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Avoid matplotlib trying to open GUI backends during the test
    monkeypatch.setenv("MPLBACKEND", "Agg")
    result = _build_result(tmp_path)

    _generate_wf_report(result, output_dir=str(tmp_path))

    expected = [
        "per_window_results_raw.csv",
        "per_window_results.csv",
        "per_window_results.md",
        "trades.csv",
        "weights_history.csv",
    ]
    for fname in expected:
        path = tmp_path / fname
        assert path.exists(), f"{fname} was not generated"
