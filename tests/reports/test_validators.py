"""Tests for report validators."""

from __future__ import annotations

from pathlib import Path

import pytest

from arara_quant.config import get_settings, reset_settings_cache
from arara_quant.reports.validators import validate_configs


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> None:
    reset_settings_cache()
    yield
    reset_settings_cache()


def test_validate_configs_returns_success_for_minimal_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True)

    (configs_dir / "universe_test.yaml").write_text(
        "\n".join(["name: TEST", "tickers:", "  - AAA"]),
        encoding="utf-8",
    )
    (configs_dir / "portfolio_test.yaml").write_text(
        "\n".join(["risk_aversion: 3.0", "max_position: 0.2"]),
        encoding="utf-8",
    )

    monkeypatch.setenv("ARARA_QUANT_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("ARARA_QUANT_CONFIGS_DIR", str(configs_dir))
    monkeypatch.setenv("ARARA_QUANT_LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("ARARA_QUANT_DATA_DIR", str(tmp_path / "data"))

    settings = get_settings()
    result = validate_configs(settings)

    assert result.is_valid()
    assert len(result.validated) == 2
    assert result.errors == ()


def test_validate_configs_collects_schema_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True)

    (configs_dir / "portfolio_bad.yaml").write_text(
        "\n".join(["max_position: 0.2", "min_position: 0.3"]),
        encoding="utf-8",
    )

    monkeypatch.setenv("ARARA_QUANT_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("ARARA_QUANT_CONFIGS_DIR", str(configs_dir))
    monkeypatch.setenv("ARARA_QUANT_LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("ARARA_QUANT_DATA_DIR", str(tmp_path / "data"))

    settings = get_settings()
    result = validate_configs(settings)

    assert not result.is_valid()
    assert len(result.errors) == 1
    assert result.errors[0][0].name == "portfolio_bad.yaml"

