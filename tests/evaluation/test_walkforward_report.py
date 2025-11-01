"""Tests for walk-forward analysis and reporting module."""

from __future__ import annotations

import pandas as pd
import pytest

from itau_quant.evaluation.walkforward_report import (
    WalkForwardSummary,
    StressPeriod,
    compute_wf_summary_stats,
    build_per_window_table,
    identify_stress_periods,
    compute_range_ratio,
    format_wf_summary_markdown,
)


@pytest.fixture
def sample_split_metrics() -> pd.DataFrame:
    """Create sample split metrics for testing."""
    return pd.DataFrame({
        "train_start": ["2020-01-01", "2020-03-01", "2020-05-01"],
        "train_end": ["2020-02-28", "2020-04-30", "2020-06-30"],
        "test_start": ["2020-03-01", "2020-05-01", "2020-07-01"],
        "test_end": ["2020-03-31", "2020-05-31", "2020-07-31"],
        "total_return": [0.05, -0.02, 0.08],
        "annualized_return": [0.25, -0.10, 0.40],
        "annualized_volatility": [0.12, 0.10, 0.15],
        "sharpe_ratio": [1.2, -0.5, 1.8],
        "max_drawdown": [-0.03, -0.12, -0.05],
        "cumulative_nav": [1.05, 1.03, 1.11],
        "turnover": [0.15, 0.20, 0.12],
        "cost_fraction": [0.0015, 0.0020, 0.0012],
    })


def test_compute_wf_summary_stats(sample_split_metrics):
    """Test computation of walk-forward summary statistics."""
    summary = compute_wf_summary_stats(sample_split_metrics)

    assert isinstance(summary, WalkForwardSummary)
    assert summary.n_windows == 3
    assert summary.success_rate == pytest.approx(2 / 3, abs=0.01)  # 2 positive out of 3
    assert summary.avg_sharpe == pytest.approx((1.2 - 0.5 + 1.8) / 3, abs=0.01)
    assert summary.avg_return == pytest.approx((0.25 - 0.10 + 0.40) / 3, abs=0.01)
    assert summary.avg_volatility == pytest.approx((0.12 + 0.10 + 0.15) / 3, abs=0.01)
    assert summary.avg_drawdown == pytest.approx((-0.03 - 0.12 - 0.05) / 3, abs=0.01)
    assert summary.avg_turnover == pytest.approx((0.15 + 0.20 + 0.12) / 3, abs=0.01)
    assert summary.avg_cost == pytest.approx((0.0015 + 0.0020 + 0.0012) / 3, abs=0.0001)
    assert summary.best_window_nav == pytest.approx(1.11, abs=0.01)
    assert summary.worst_window_nav == pytest.approx(1.03, abs=0.01)
    assert summary.range_ratio == pytest.approx(1.11 / 1.03, abs=0.01)


def test_compute_wf_summary_stats_empty():
    """Test that empty split_metrics raises ValueError."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="split_metrics is empty"):
        compute_wf_summary_stats(empty_df)


def test_compute_wf_summary_stats_missing_columns():
    """Test that missing required columns raises ValueError."""
    incomplete_df = pd.DataFrame({
        "total_return": [0.05, 0.08],
        "sharpe_ratio": [1.2, 1.8],
    })
    with pytest.raises(ValueError, match="Missing required columns"):
        compute_wf_summary_stats(incomplete_df)


def test_build_per_window_table_markdown(sample_split_metrics):
    """Test per-window table generation in markdown format."""
    table = build_per_window_table(sample_split_metrics, format="markdown")

    assert isinstance(table, str)
    assert "Window End" in table
    assert "Sharpe (OOS)" in table
    assert "2020-03-31" in table
    assert "2020-05-31" in table
    assert "2020-07-31" in table


def test_build_per_window_table_csv(sample_split_metrics):
    """Test per-window table generation in CSV format."""
    table = build_per_window_table(sample_split_metrics, format="csv")

    assert isinstance(table, str)
    assert "Window End" in table
    assert "Sharpe (OOS)" in table
    assert "2020-03-31" in table


def test_build_per_window_table_latex(sample_split_metrics):
    """Test per-window table generation in LaTeX format."""
    table = build_per_window_table(sample_split_metrics, format="latex")

    assert isinstance(table, str)
    assert "\\begin{tabular}" in table
    assert "Window End" in table


def test_build_per_window_table_unsupported_format(sample_split_metrics):
    """Test that unsupported format raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported format"):
        build_per_window_table(sample_split_metrics, format="html")


def test_identify_stress_periods(sample_split_metrics):
    """Test identification of stress periods."""
    stress_periods = identify_stress_periods(
        sample_split_metrics,
        drawdown_threshold=-0.10,
        sharpe_threshold=-0.3,
    )

    assert isinstance(stress_periods, list)
    assert len(stress_periods) == 1  # Only the second window (drawdown=-0.12, sharpe=-0.5)

    period = stress_periods[0]
    assert isinstance(period, StressPeriod)
    assert period.test_start == "2020-05-01"
    assert period.test_end == "2020-05-31"
    assert period.sharpe == pytest.approx(-0.5, abs=0.01)
    assert period.max_drawdown == pytest.approx(-0.12, abs=0.01)
    assert "2020" in period.label


def test_identify_stress_periods_empty():
    """Test that empty split_metrics returns empty list."""
    empty_df = pd.DataFrame()
    stress_periods = identify_stress_periods(empty_df)
    assert stress_periods == []


def test_identify_stress_periods_none_found():
    """Test when no stress periods meet thresholds."""
    good_metrics = pd.DataFrame({
        "test_start": ["2020-01-01"],
        "test_end": ["2020-01-31"],
        "sharpe_ratio": [1.5],
        "max_drawdown": [-0.02],
        "annualized_return": [0.20],
    })
    stress_periods = identify_stress_periods(
        good_metrics,
        drawdown_threshold=-0.10,
        sharpe_threshold=-0.5,
    )
    assert stress_periods == []


def test_compute_range_ratio(sample_split_metrics):
    """Test range ratio computation."""
    ratio_stats = compute_range_ratio(sample_split_metrics)

    assert isinstance(ratio_stats, dict)
    assert "best_nav" in ratio_stats
    assert "worst_nav" in ratio_stats
    assert "range_ratio" in ratio_stats
    assert "nav_std" in ratio_stats

    assert ratio_stats["best_nav"] == pytest.approx(1.11, abs=0.01)
    assert ratio_stats["worst_nav"] == pytest.approx(1.03, abs=0.01)
    assert ratio_stats["range_ratio"] == pytest.approx(1.11 / 1.03, abs=0.01)
    assert ratio_stats["nav_std"] > 0


def test_compute_range_ratio_empty():
    """Test range ratio with empty DataFrame returns NaN."""
    empty_df = pd.DataFrame()
    ratio_stats = compute_range_ratio(empty_df)

    assert ratio_stats["best_nav"] is pd.NA or pd.isna(ratio_stats["best_nav"])
    assert ratio_stats["worst_nav"] is pd.NA or pd.isna(ratio_stats["worst_nav"])


def test_format_wf_summary_markdown():
    """Test markdown formatting of summary statistics."""
    summary = WalkForwardSummary(
        n_windows=5,
        success_rate=0.8,
        avg_sharpe=1.2,
        avg_return=0.15,
        avg_volatility=0.12,
        avg_drawdown=-0.08,
        avg_turnover=0.10,
        avg_cost=0.0015,
        consistency_r2=0.65,
        best_window_nav=1.25,
        worst_window_nav=1.05,
        range_ratio=1.19,
    )

    markdown = format_wf_summary_markdown(summary)

    assert isinstance(markdown, str)
    assert "Walk-Forward Summary Statistics" in markdown
    assert "Number of OOS Windows" in markdown
    assert "5" in markdown
    assert "80.0%" in markdown  # success_rate
    assert "1.2" in markdown or "1.20" in markdown  # avg_sharpe
    assert "0.65" in markdown or "0.650" in markdown  # consistency_r2


def test_consistency_r2_single_window():
    """Test consistency R² with only 1 window returns 0."""
    single_window = pd.DataFrame({
        "train_start": ["2020-01-01"],
        "train_end": ["2020-02-28"],
        "test_start": ["2020-03-01"],
        "test_end": ["2020-03-31"],
        "total_return": [0.05],
        "annualized_return": [0.25],
        "annualized_volatility": [0.12],
        "sharpe_ratio": [1.2],
        "max_drawdown": [-0.03],
        "cumulative_nav": [1.05],
        "turnover": [0.15],
        "cost_fraction": [0.0015],
    })

    summary = compute_wf_summary_stats(single_window)
    assert summary.consistency_r2 == 0.0
