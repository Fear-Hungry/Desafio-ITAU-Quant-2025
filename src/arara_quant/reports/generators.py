"""Report and artefact generation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from arara_quant.config import Settings, get_settings
from arara_quant.reports.canonical import load_oos_period
from arara_quant.utils.yaml_loader import read_yaml

__all__ = ["TurnoverStats", "update_readme_turnover_stats"]


@dataclass(frozen=True, slots=True)
class TurnoverStats:
    median: float
    p95: float


def _format_sci(value: float) -> str:
    return f"{float(value):.2e}"


def _load_turnover_summary(path: Path) -> dict[str, TurnoverStats]:
    if not path.exists():
        return {}

    frame = pd.read_csv(path)
    required = {"strategy", "turnover_median", "turnover_p95"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"turnover summary missing columns {sorted(missing)} in {path}"
        )

    result: dict[str, TurnoverStats] = {}
    for _, row in frame.iterrows():
        key = str(row["strategy"]).strip()
        result[key] = TurnoverStats(
            median=float(row["turnover_median"]),
            p95=float(row["turnover_p95"]),
        )
    return result


def _compute_turnover_from_trades(
    trades_csv: Path, *, period_start: pd.Timestamp, period_end: pd.Timestamp
) -> TurnoverStats | None:
    if not trades_csv.exists():
        return None

    frame = pd.read_csv(trades_csv)
    if "date" not in frame.columns or "turnover" not in frame.columns:
        return None

    frame["date"] = pd.to_datetime(frame["date"])
    mask = (frame["date"] >= period_start) & (frame["date"] <= period_end)
    values = pd.to_numeric(frame.loc[mask, "turnover"], errors="coerce").dropna()
    if values.empty:
        return None
    return TurnoverStats(median=float(values.median()), p95=float(values.quantile(0.95)))


def _compute_turnover_from_windows(
    per_window_csv: Path, *, period_start: pd.Timestamp, period_end: pd.Timestamp
) -> TurnoverStats | None:
    if not per_window_csv.exists():
        return None

    frame = pd.read_csv(per_window_csv)
    end_col = "Window End" if "Window End" in frame.columns else "date"
    turn_col = "Turnover" if "Turnover" in frame.columns else "turnover"
    if end_col not in frame.columns or turn_col not in frame.columns:
        return None

    frame[end_col] = pd.to_datetime(frame[end_col])
    mask = (frame[end_col] >= period_start) & (frame[end_col] <= period_end)
    values = pd.to_numeric(frame.loc[mask, turn_col], errors="coerce").dropna()
    if values.empty:
        return None
    return TurnoverStats(median=float(values.median()), p95=float(values.quantile(0.95)))


def _find_markdown_table(lines: list[str]) -> tuple[int, int]:
    """Locate the markdown table block for README table 5.1."""

    start = -1
    for idx, line in enumerate(lines):
        header = line.strip()
        if not (header.startswith("| Estratégia") or header.lower().startswith("| strategy")):
            continue
        lower = header.lower()
        has_turnover = "turnover" in lower
        has_med = ("mediana" in lower) or ("mediano" in lower) or ("median" in lower)
        has_p95 = "p95" in lower
        if has_turnover and has_med and has_p95:
            start = idx
            break

    if start == -1:
        raise RuntimeError(
            "Could not find README markdown table header containing turnover mediana and p95 columns"
        )

    end = start + 1
    while end < len(lines) and lines[end].strip().startswith("|"):
        end += 1

    return start, end


def _parse_table(table_lines: list[str]) -> tuple[list[str], list[list[str]]]:
    if len(table_lines) < 2:
        raise ValueError("Table too short to parse")

    header = [c.strip() for c in table_lines[0].strip().strip("|").split("|")]
    rows: list[list[str]] = []
    for line in table_lines[2:]:
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < len(header):
            cells += [""] * (len(header) - len(cells))
        elif len(cells) > len(header):
            cells = cells[: len(header)]
        rows.append(cells)
    return header, rows


def _rebuild_table(
    header: list[str], rows: list[list[str]], *, separator_line: str | None
) -> list[str]:
    lines = ["| " + " | ".join(header) + " |"]
    if separator_line and separator_line.strip().startswith("|"):
        lines.append(separator_line.rstrip("\n"))
    else:
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _find_column(header: list[str], *, parts: list[str]) -> int:
    parts_lower = [p.lower() for p in parts]
    for idx, col in enumerate(header):
        text = col.lower()
        if all(part in text for part in parts_lower):
            return idx
    raise ValueError(f"Column {parts} not found in header: {header}")


def update_readme_turnover_stats(
    *,
    settings: Settings | None = None,
    readme_path: Path | None = None,
    baseline_summary_csv: Path | None = None,
    prism_per_window_csv: Path | None = None,
    prism_trades_csv: Path | None = None,
    oos_config_path: Path | None = None,
    force_overwrite: bool = False,
) -> int:
    """Update README Table 5.1 turnover (median/p95) columns in-place.

    Returns the number of strategies updated.
    """

    settings = settings or get_settings()

    readme_path = readme_path or (settings.project_root / "README.md")
    baseline_summary_csv = baseline_summary_csv or (
        settings.results_dir / "oos_canonical" / "turnover_dist_stats.csv"
    )
    prism_per_window_csv = prism_per_window_csv or (
        settings.walkforward_dir / "per_window_results.csv"
    )
    prism_trades_csv = prism_trades_csv or (settings.walkforward_dir / "trades.csv")

    if oos_config_path is not None:
        payload = read_yaml(oos_config_path)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid YAML root mapping in {oos_config_path}")
        oos = payload.get("oos_evaluation") or {}
        if not isinstance(oos, dict):
            raise ValueError(f"Missing oos_evaluation mapping in {oos_config_path}")
        start = pd.to_datetime(oos.get("start_date"))
        end = pd.to_datetime(oos.get("end_date"))
        if pd.isna(start) or pd.isna(end):
            raise ValueError(f"Invalid OOS period in {oos_config_path}")
        period_start = start
        period_end = end
    else:
        period = load_oos_period(settings)
        period_start = period.start
        period_end = period.end
    baseline_stats = _load_turnover_summary(baseline_summary_csv)

    prism_stats = _compute_turnover_from_trades(
        prism_trades_csv, period_start=period_start, period_end=period_end
    ) or _compute_turnover_from_windows(
        prism_per_window_csv, period_start=period_start, period_end=period_end
    )

    lines = readme_path.read_text(encoding="utf-8").splitlines()
    start, end = _find_markdown_table(lines)
    table_lines = lines[start:end]
    header, rows = _parse_table(table_lines)

    col_med = _find_column(header, parts=["turnover", "med"])
    col_p95 = _find_column(header, parts=["turnover", "p95"])
    try:
        col_strategy = _find_column(header, parts=["strategy"])
    except ValueError:
        col_strategy = _find_column(header, parts=["estratégia"])

    updated = 0
    for row in rows:
        strategy_cell = row[col_strategy]
        key = strategy_cell.strip()

        replacement: TurnoverStats | None = None
        if "PRISM" in key.upper() and prism_stats is not None:
            replacement = prism_stats
        else:
            replacement = baseline_stats.get(key)

        if replacement is None:
            continue

        med_value = _format_sci(replacement.median)
        p95_value = _format_sci(replacement.p95)

        def _should_write(cell: str) -> bool:
            if force_overwrite:
                return True
            stripped = cell.strip()
            return stripped in {"", "—", "-", "–"}

        wrote = False
        if _should_write(row[col_med]):
            row[col_med] = med_value
            wrote = True
        if _should_write(row[col_p95]):
            row[col_p95] = p95_value
            wrote = True
        if wrote:
            updated += 1

    separator = table_lines[1] if len(table_lines) > 1 else None
    rebuilt = _rebuild_table(header, rows, separator_line=separator)
    lines[start:end] = rebuilt
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return updated
