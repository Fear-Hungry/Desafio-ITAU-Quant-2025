from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from arara_quant.config import get_settings, reset_settings_cache
from arara_quant.reports.canonical import (
    ensure_output_dirs,
    load_nav_daily,
    load_oos_config,
    load_oos_period,
    subset_to_oos_period,
)


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> None:
    reset_settings_cache()
    yield
    reset_settings_cache()


def test_load_oos_period_and_subset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = tmp_path
    configs_dir = project_root / "configs"
    reports_dir = project_root / "outputs" / "reports"
    nav_path = reports_dir / "walkforward" / "nav_daily.csv"

    configs_dir.mkdir(parents=True, exist_ok=True)
    nav_path.parent.mkdir(parents=True, exist_ok=True)

    (configs_dir / "oos_period.yaml").write_text(
        "\n".join(
            [
                "oos_evaluation:",
                '  start_date: "2020-01-02"',
                '  end_date: "2020-01-03"',
            ]
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        {
            "date": ["2020-01-03", "2020-01-02", "2020-01-06"],
            "nav": [1.01, 1.0, 0.99],
        }
    ).to_csv(nav_path, index=False)

    monkeypatch.setenv("ARARA_QUANT_PROJECT_ROOT", str(project_root))
    monkeypatch.setenv("ARARA_QUANT_CONFIGS_DIR", str(configs_dir))
    monkeypatch.setenv("ARARA_QUANT_REPORTS_DIR", str(reports_dir))

    settings = get_settings()
    ensure_output_dirs(settings)

    cfg = load_oos_config(settings)
    assert cfg["start_date"] == "2020-01-02"
    assert cfg["end_date"] == "2020-01-03"

    period = load_oos_period(settings)
    assert period.start == pd.Timestamp("2020-01-02")
    assert period.end == pd.Timestamp("2020-01-03")

    nav = load_nav_daily(settings)
    assert nav["date"].is_monotonic_increasing

    subset = subset_to_oos_period(nav, period)
    assert subset["date"].min() == pd.Timestamp("2020-01-02")
    assert subset["date"].max() == pd.Timestamp("2020-01-03")
