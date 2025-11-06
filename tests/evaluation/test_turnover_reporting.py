from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.update_readme_turnover_stats import load_oos_period


def _extract_first_match(text: str, label: str) -> float:
    pattern = re.compile(
        rf"\|\s*{re.escape(label)}\s*\|\s*([0-9eE\.\-]+)",
        flags=re.IGNORECASE,
    )
    match = pattern.search(text)
    if not match:
        raise AssertionError(f"Could not locate '{label}' entry in README")
    return float(match.group(1))


def test_per_window_turnover_matches_readme_summary() -> None:
    """README turnover stats must match per-window CSV derived metrics."""

    per_window = Path("reports/walkforward/per_window_results.csv")
    readme = Path("README.md")

    assert per_window.exists(), "Expected per-window results CSV"
    assert readme.exists(), "Expected README.md to exist"

    oos = load_oos_period(Path("configs/oos_period.yaml"))
    df = pd.read_csv(per_window, parse_dates=["Window End"])
    mask = (df["Window End"] >= oos.start) & (df["Window End"] <= oos.end)
    filtered = df.loc[mask]
    assert not filtered.empty, "Filtered per-window dataframe is empty"

    turnover_median = float(filtered["Turnover"].median())
    turnover_p95 = float(filtered["Turnover"].quantile(0.95))
    cost_mean = float(filtered["Cost"].mean())
    cost_annual_bps = cost_mean * 252 * 10_000

    readme_text = readme.read_text(encoding="utf-8")

    observed_turnover_median = _extract_first_match(
        readme_text, "Turnover mediano (‖Δw‖₁ one-way)"
    )
    observed_turnover_p95 = _extract_first_match(readme_text, "Turnover p95")
    observed_cost_mean = _extract_first_match(readme_text, "Custo médio por rebalance")
    observed_cost_annual = _extract_first_match(readme_text, "Custo anualizado (bps)")

    assert abs(observed_turnover_median - turnover_median) < 1e-12
    assert abs(observed_turnover_p95 - turnover_p95) < 1e-12
    assert abs(observed_cost_mean - cost_mean) < 1e-12
    assert abs(observed_cost_annual - cost_annual_bps) < 1e-9
