"""Canonical artefact loaders for scripts and reports.

This module centralises file locations and parsing for the "single source of truth"
artefacts referenced throughout the repository (configs + outputs).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from arara_quant.config import Settings, get_settings
from arara_quant.utils.yaml_loader import read_yaml

__all__ = [
    "OOSPeriod",
    "ensure_output_dirs",
    "load_baseline_metrics_oos",
    "load_nav_daily",
    "load_oos_consolidated_metrics",
    "load_oos_config",
    "load_oos_period",
    "load_walkforward_windows",
    "resolve_baseline_metrics_path",
    "resolve_consolidated_metrics_path",
    "resolve_nav_daily_path",
    "resolve_oos_config_path",
    "resolve_walkforward_windows_path",
    "subset_to_oos_period",
]


@dataclass(frozen=True, slots=True)
class OOSPeriod:
    start: pd.Timestamp
    end: pd.Timestamp


def _resolve_settings(settings: Settings | None) -> Settings:
    return settings or get_settings()


def ensure_output_dirs(settings: Settings | None = None) -> None:
    settings = _resolve_settings(settings)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    settings.walkforward_dir.mkdir(parents=True, exist_ok=True)
    settings.figures_dir.mkdir(parents=True, exist_ok=True)


def resolve_oos_config_path(settings: Settings | None = None) -> Path:
    settings = _resolve_settings(settings)
    return settings.configs_dir / "oos_period.yaml"


def resolve_nav_daily_path(settings: Settings | None = None) -> Path:
    settings = _resolve_settings(settings)
    return settings.walkforward_dir / "nav_daily.csv"


def resolve_walkforward_windows_path(settings: Settings | None = None) -> Path:
    settings = _resolve_settings(settings)
    return settings.walkforward_dir / "per_window_results.csv"


def resolve_consolidated_metrics_path(settings: Settings | None = None) -> Path:
    settings = _resolve_settings(settings)
    return settings.reports_dir / "oos_consolidated_metrics.json"


def resolve_baseline_metrics_path(settings: Settings | None = None) -> Path:
    settings = _resolve_settings(settings)
    return settings.results_dir / "baselines" / "baseline_metrics_oos.csv"


def _require_mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping, got {type(value).__name__}")
    return value


def load_oos_config(settings: Settings | None = None) -> Mapping[str, Any]:
    """Load the canonical OOS evaluation config mapping."""

    path = resolve_oos_config_path(settings)
    data = read_yaml(path)
    mapping = _require_mapping(data, context=f"YAML root in {path}")
    oos = mapping.get("oos_evaluation")
    return _require_mapping(oos, context=f"'oos_evaluation' in {path}")


def load_oos_period(settings: Settings | None = None) -> OOSPeriod:
    """Return the canonical OOS period parsed as timestamps."""

    cfg = load_oos_config(settings)
    start_raw = cfg.get("start_date")
    end_raw = cfg.get("end_date")
    start = pd.to_datetime(start_raw)
    end = pd.to_datetime(end_raw)
    if pd.isna(start) or pd.isna(end):
        raise ValueError("Invalid OOS period in oos_period.yaml")
    return OOSPeriod(start=start, end=end)


def load_nav_daily(settings: Settings | None = None) -> pd.DataFrame:
    """Load the canonical daily NAV series (single source of truth)."""

    path = resolve_nav_daily_path(settings)
    frame = pd.read_csv(path)
    if "date" not in frame.columns:
        raise ValueError(f"nav_daily.csv missing 'date' column: {path}")
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)
    return frame


def load_walkforward_windows(settings: Settings | None = None) -> pd.DataFrame:
    """Load the canonical per-window results table exported by the CLI."""

    path = resolve_walkforward_windows_path(settings)
    frame = pd.read_csv(path)
    for candidate in ("Window End", "date"):
        if candidate in frame.columns:
            frame[candidate] = pd.to_datetime(frame[candidate])
            break
    return frame


def subset_to_oos_period(
    frame: pd.DataFrame,
    period: OOSPeriod,
    *,
    date_column: str = "date",
) -> pd.DataFrame:
    """Filter a frame to the inclusive OOS period based on a date column."""

    if date_column not in frame.columns:
        raise ValueError(f"Missing '{date_column}' column for OOS filtering.")
    dates = pd.to_datetime(frame[date_column])
    mask = (dates >= period.start) & (dates <= period.end)
    return frame.loc[mask].copy()


def load_oos_consolidated_metrics(settings: Settings | None = None) -> dict[str, Any]:
    path = resolve_consolidated_metrics_path(settings)
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def load_baseline_metrics_oos(settings: Settings | None = None) -> pd.DataFrame:
    path = resolve_baseline_metrics_path(settings)
    return pd.read_csv(path)
