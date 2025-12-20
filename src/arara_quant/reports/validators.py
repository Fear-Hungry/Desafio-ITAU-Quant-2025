"""Validation helpers for report-generation pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from arara_quant.config import (
    PortfolioConfig,
    Settings,
    UniverseConfig,
    get_settings,
    load_config,
)

__all__ = ["ConfigValidationResult", "validate_configs"]


@dataclass(frozen=True, slots=True)
class ConfigValidationResult:
    """Result of validating YAML configs under ``configs/``."""

    validated: tuple[Path, ...]
    errors: tuple[tuple[Path, str], ...]

    def is_valid(self) -> bool:
        return len(self.errors) == 0


def validate_configs(settings: Settings | None = None) -> ConfigValidationResult:
    """Validate all YAML configuration files under ``settings.configs_dir``.

    Validates:
    - ``universe_*.yaml`` against :class:`UniverseConfig`
    - ``portfolio_*.yaml`` against :class:`PortfolioConfig`
    """

    settings = settings or get_settings()
    configs_dir = settings.configs_dir

    validated: list[Path] = []
    errors: list[tuple[Path, str]] = []

    for config_file in sorted(configs_dir.glob("universe_*.yaml")):
        validated.append(config_file)
        try:
            load_config(str(config_file), UniverseConfig)
        except Exception as exc:  # pragma: no cover - thin wrapper
            errors.append((config_file, str(exc)))

    for config_file in sorted(configs_dir.glob("portfolio_*.yaml")):
        validated.append(config_file)
        try:
            load_config(str(config_file), PortfolioConfig)
        except Exception as exc:  # pragma: no cover - thin wrapper
            errors.append((config_file, str(exc)))

    return ConfigValidationResult(
        validated=tuple(validated),
        errors=tuple(errors),
    )

