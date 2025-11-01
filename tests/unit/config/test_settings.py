from __future__ import annotations

from pathlib import Path

import pytest
from itau_quant.config.settings import Settings, load_env_file, reset_settings_cache


def teardown_module() -> None:  # pragma: no cover - test helper
    reset_settings_cache()


def test_load_env_file_parses_key_value(tmp_path: Path) -> None:
    env_path = tmp_path / "custom.env"
    env_path.write_text(
        """
        # comment
        ITAU_QUANT_ENVIRONMENT=production
        ITAU_QUANT_RANDOM_SEED=101
        INVALID_LINE
        """.strip(),
        encoding="utf-8",
    )

    data = load_env_file(env_path)
    assert data["ITAU_QUANT_ENVIRONMENT"] == "production"
    assert data["ITAU_QUANT_RANDOM_SEED"] == "101"
    assert "INVALID_LINE" not in data


def test_settings_from_env_uses_project_root_and_env_file(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "ITAU_QUANT_RANDOM_SEED=123\nITAU_QUANT_STRUCTURED_LOGGING=false\n",
        encoding="utf-8",
    )

    settings = Settings.from_env(overrides={"project_root": tmp_path})

    assert settings.project_root == tmp_path.resolve()
    assert settings.random_seed == 123
    assert settings.data_dir == tmp_path / "data"
    assert settings.raw_data_dir == tmp_path / "data" / "raw"
    assert not settings.structured_logging


def test_settings_from_env_overrides_and_env_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ITAU_QUANT_DATA_DIR", "env_data")
    monkeypatch.setenv("ITAU_QUANT_STRUCTURED_LOGGING", "1")

    settings = Settings.from_env(
        overrides={
            "project_root": tmp_path,
            "LOGS_DIR": "logs_alt",
            "STRUCTURED_LOGGING": "false",
        }
    )

    assert settings.data_dir == (tmp_path / "env_data")
    assert settings.logs_dir == (tmp_path / "logs_alt")
    # Overrides take precedence over environment variables.
    assert not settings.structured_logging


def test_settings_unknown_override_raises(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        Settings.from_env(overrides={"project_root": tmp_path, "UNKNOWN": 10})
