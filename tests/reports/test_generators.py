"""Tests for report generators."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from arara_quant.config import get_settings, reset_settings_cache
from arara_quant.reports.generators import update_readme_turnover_stats


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> None:
    reset_settings_cache()
    yield
    reset_settings_cache()


def test_update_readme_turnover_stats_updates_placeholders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = tmp_path
    configs_dir = project_root / "configs"
    results_dir = project_root / "outputs" / "results"
    reports_dir = project_root / "outputs" / "reports"
    walkforward_dir = reports_dir / "walkforward"

    configs_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "oos_canonical").mkdir(parents=True, exist_ok=True)
    walkforward_dir.mkdir(parents=True, exist_ok=True)

    # Settings / OOS period
    (configs_dir / "oos_period.yaml").write_text(
        "\n".join(
            [
                "oos_evaluation:",
                '  start_date: "2020-01-01"',
                '  end_date: "2020-01-31"',
            ]
        ),
        encoding="utf-8",
    )

    # README with a minimal table
    readme_path = project_root / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# README",
                "",
                "| Estratégia | Turnover (mediana) | Turnover (p95) |",
                "|---|---:|---:|",
                "| PRISM-R (Portfolio Optimization) | — | — |",
                "| equal_weight | — | — |",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # Baseline summary CSV
    baseline_csv = results_dir / "oos_canonical" / "turnover_dist_stats.csv"
    pd.DataFrame(
        [
            {"strategy": "equal_weight", "turnover_median": 0.01, "turnover_p95": 0.02},
        ]
    ).to_csv(baseline_csv, index=False)

    # PRISM per-window CSV (used when trades.csv missing)
    per_window_csv = walkforward_dir / "per_window_results.csv"
    window_turnover = pd.Series([0.10, 0.15, 0.05], dtype=float)
    pd.DataFrame(
        {
            "Window End": ["2020-01-10", "2020-01-20", "2020-02-01"],
            "Turnover": window_turnover,
        }
    ).to_csv(per_window_csv, index=False)

    monkeypatch.setenv("ARARA_QUANT_PROJECT_ROOT", str(project_root))
    monkeypatch.setenv("ARARA_QUANT_CONFIGS_DIR", str(configs_dir))
    monkeypatch.setenv("ARARA_QUANT_RESULTS_DIR", str(results_dir))
    monkeypatch.setenv("ARARA_QUANT_REPORTS_DIR", str(reports_dir))

    settings = get_settings()

    updated = update_readme_turnover_stats(settings=settings)
    assert updated == 2

    updated_text = readme_path.read_text(encoding="utf-8")

    prism_oos = window_turnover.iloc[:2]
    prism_expected_median = float(prism_oos.median())
    prism_expected_p95 = float(prism_oos.quantile(0.95))
    assert f"{prism_expected_median:.2e}" in updated_text
    assert f"{prism_expected_p95:.2e}" in updated_text
    assert "1.00e-02" in updated_text
    assert "2.00e-02" in updated_text
