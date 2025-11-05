from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from arara_quant.cli import main
from arara_quant.config import reset_settings_cache


def setup_function() -> None:  # pragma: no cover - helper
    reset_settings_cache()


def teardown_function() -> None:  # pragma: no cover - helper
    reset_settings_cache()


def test_show_settings_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ARARA_QUANT_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("ARARA_QUANT_LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("ARARA_QUANT_CONFIGS_DIR", str(tmp_path / "configs"))
    monkeypatch.setenv("ARARA_QUANT_DATA_DIR", str(tmp_path / "data"))

    exit_code = main(["show-settings", "--json"])
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert Path(payload["project_root"]) == tmp_path.resolve()
    assert payload["logs_dir"].endswith("logs")


def test_optimize_resolves_default_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True)
    (configs_dir / "optimizer_example.yaml").write_text(
        "optimizer:\n  lambda: 3.0\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("ARARA_QUANT_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("ARARA_QUANT_CONFIGS_DIR", str(configs_dir))
    monkeypatch.setenv("ARARA_QUANT_LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("ARARA_QUANT_DATA_DIR", str(tmp_path / "data"))

    exit_code = main(["optimize", "--json"])
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["config_path"].endswith("optimizer_example.yaml")
    assert payload["status"] == "preview"
    assert payload["dry_run"] is True


def test_backtest_alternate_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True)
    custom_config = configs_dir / "custom.yaml"
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)
    returns = pd.DataFrame(
        {
            "AAA": np.full(120, 0.0005),
            "BBB": np.linspace(0.0001, 0.0008, 120),
        },
        index=pd.bdate_range("2021-01-01", periods=120),
    )
    returns_path = data_dir / "returns_cli.csv"
    returns.to_csv(returns_path)

    custom_config.write_text(
        f"""
base_currency: USD
data:
  returns: {returns_path}
walkforward:
  train_days: 40
  test_days: 20
optimizer:
  tau: 0.25
estimators:
  costs:
    linear_bps: 5
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("ARARA_QUANT_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("ARARA_QUANT_CONFIGS_DIR", str(configs_dir))
    monkeypatch.setenv("ARARA_QUANT_LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("ARARA_QUANT_DATA_DIR", str(tmp_path / "data"))

    exit_code = main(
        ["backtest", "--config", str(custom_config), "--json", "--no-dry-run"]
    )
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert Path(payload["config_path"]) == custom_config.resolve()
    assert payload["status"] == "completed"
    assert payload["metrics"]["total_return"] > 0
