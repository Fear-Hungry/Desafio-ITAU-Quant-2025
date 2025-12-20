#!/usr/bin/env python3
"""
Export per-window (per-rebalance) turnover for baseline strategies.

This script computes and writes per-window turnover series for:
- equal_weight      → Equal-weight portfolio over the universe (one-way turnover)
- sixty_forty       → 60% equities (SPY), 40% bonds (TLT) with monthly rebalance
- risk_parity       → Placeholder using inverse-volatility parity (IVP) over the universe

Definitions
- Rebalance schedule: first business day of each month in the OOS period (or custom start/end)
- Turnover (one-way): 0.5 * Σ_i |w_target_i - w_pretrade_i|, where w_pretrade_i são os
  pesos após drift desde o último rebalance (isto é, pesos após evolução com retornos).

Outputs
- outputs/results/oos_canonical/per_window/turnover_<strategy>.csv → columns: date, turnover
- outputs/results/oos_canonical/turnover_dist_stats.csv            → median, p95 por estratégia

Defaults
- Período OOS: carregado de configs/oos_period.yaml (start_date, end_date)
- Dados: tenta data/processed/returns_arara.parquet por padrão (pode ser custom via --returns)
- Universo: interseção entre universe_arara.yaml (tickers) e colunas do DataFrame de retornos

Usage
    poetry run python scripts/baselines/export_per_window_turnover.py \
        --strategies equal_weight sixty_forty risk_parity

Requisitos
- pandas, numpy, pyyaml
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIGS = REPO_ROOT / "configs"
DEFAULT_REPORTS = REPO_ROOT / "outputs" / "reports"
DEFAULT_RESULTS = REPO_ROOT / "outputs" / "results"
DEFAULT_DATA = REPO_ROOT / "data" / "processed"

DEFAULT_OOS_CONFIG = DEFAULT_CONFIGS / "oos_period.yaml"
DEFAULT_UNIVERSE = DEFAULT_CONFIGS / "universe_arara.yaml"

# Fallbacks para returns
DEFAULT_RETURNS_CANDIDATES = [
    DEFAULT_DATA / "returns_arara.parquet",
    DEFAULT_DATA / "returns.parquet",
    DEFAULT_DATA / "returns.csv",
]


@dataclass
class OOSPeriod:
    start: pd.Timestamp
    end: pd.Timestamp


def load_oos_period(config_path: Path) -> OOSPeriod:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    oos = cfg.get("oos_evaluation", {})
    start = pd.to_datetime(oos.get("start_date"))
    end = pd.to_datetime(oos.get("end_date"))
    if pd.isna(start) or pd.isna(end):
        raise ValueError(f"Invalid OOS period in {config_path}")
    return OOSPeriod(start=start, end=end)


def load_universe(universe_path: Path) -> List[str]:
    with open(universe_path, "r", encoding="utf-8") as f:
        u = yaml.safe_load(f)
    tickers = u.get("tickers", [])
    # Normalize to upper-case strings
    return [str(t).upper() for t in tickers]


def autodetect_returns(path_candidates: Iterable[Path]) -> Path:
    for p in path_candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find returns file. Tried: {', '.join(str(p) for p in path_candidates)}"
    )


def load_returns(path: Path) -> pd.DataFrame:
    """Load returns DataFrame with DatetimeIndex and columns=tickers, values=decimal returns."""
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() in (".csv", ".txt"):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported returns file type: {path}")

    # Try to coerce date index
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    elif "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    elif not isinstance(df.index, pd.DatetimeIndex):
        # Best-effort: try first column
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            raise ValueError("Returns file must have a datetime index or a 'date' column") from e

    # Ensure numeric
    df = df.sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    # Drop all-nan columns
    df = df.dropna(axis=1, how="all")
    return df


def first_business_days(index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    """Return the first available business day in index for each month between start and end."""
    # Monthly period starts (Month Start)
    months = pd.date_range(start=start.normalize(), end=end.normalize(), freq="MS")
    fbd = []
    for m in months:
        # find first index date >= m
        i = index.searchsorted(m)
        if i < len(index):
            d = index[i]
            if d <= end:
                fbd.append(d)
    # Deduplicate in case of identical months with missing dates
    uniq = []
    last = None
    for d in fbd:
        if (last is None) or (d != last):
            uniq.append(d)
            last = d
    return uniq


def compute_pretrade_weights(w_prev: pd.Series, rets: pd.DataFrame) -> pd.Series:
    """
    Drift previous weights with realized returns between rebalances:
      w_pre ∝ w_prev * Π_{t in (prev,next)} (1 + r_t)
    """
    if w_prev.empty or rets.empty:
        return w_prev.copy()

    growth = (1.0 + rets).prod(axis=0)
    # Keep only tickers in previous weights
    growth = growth.reindex(w_prev.index).fillna(1.0)
    w_pre = w_prev * growth
    total = w_pre.sum()
    if total <= 0 or not np.isfinite(total):
        # Fallback: normalize equally if degenerate
        n = (w_prev > 0).sum()
        if n == 0:
            return w_prev
        return pd.Series(np.full(len(w_prev), 1.0 / len(w_prev)), index=w_prev.index)
    return (w_pre / total).fillna(0.0)


def turnover_one_way(w_pre: pd.Series, w_tgt: pd.Series) -> float:
    """One-way turnover: 0.5 * sum |w_tgt - w_pre| over intersecting tickers."""
    # Align indices
    idx = sorted(set(w_pre.index).union(w_tgt.index))
    a = w_pre.reindex(idx, fill_value=0.0).values
    b = w_tgt.reindex(idx, fill_value=0.0).values
    return 0.5 * float(np.abs(b - a).sum())


def target_weights_equal_weight(tickers: List[str]) -> pd.Series:
    n = len(tickers)
    if n == 0:
        return pd.Series(dtype=float)
    w = np.full(n, 1.0 / n, dtype=float)
    return pd.Series(w, index=tickers)


def target_weights_sixty_forty(all_tickers: List[str]) -> pd.Series:
    """
    60% SPY, 40% TLT.
    Fallbacks: try SPY/AGG if TLT not present. Raise if neither bond proxy found.
    """
    cols = set(all_tickers)
    if "SPY" not in cols:
        raise ValueError("60/40 requires SPY in returns columns")
    bond = None
    if "TLT" in cols:
        bond = "TLT"
    elif "AGG" in cols:
        bond = "AGG"
    else:
        raise ValueError("60/40 requires TLT or AGG in returns columns")
    return pd.Series({"SPY": 0.60, bond: 0.40})


def target_weights_inverse_vol(rets_window: pd.DataFrame, floor: float = 1e-8) -> pd.Series:
    """
    Placeholder for Risk Parity: inverse-volatility parity (diagonal RP approximation).
    Weights ∝ 1 / vol, normalized to 1. Uses sample std dev over the lookback window.
    """
    if rets_window.empty:
        return pd.Series(dtype=float)
    vol = rets_window.std(ddof=1).replace(0.0, np.nan)
    iv = 1.0 / vol
    iv = iv.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if iv.sum() <= floor:
        # fallback equal-weight
        return target_weights_equal_weight(list(rets_window.columns))
    w = iv / iv.sum()
    return w


def filter_universe(df: pd.DataFrame, tickers: Optional[List[str]]) -> pd.DataFrame:
    if not tickers:
        return df
    cols = [c for c in tickers if c in df.columns]
    return df[cols].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export per-window turnover for baselines")
    parser.add_argument(
        "--returns",
        type=str,
        default=None,
        help="Path to returns file (parquet or csv). Default: data/processed/returns_arara.parquet",
    )
    parser.add_argument(
        "--universe",
        type=str,
        default=str(DEFAULT_UNIVERSE),
        help="Universe YAML (tickers list). Default: configs/universe_arara.yaml",
    )
    parser.add_argument(
        "--oos-config",
        type=str,
        default=str(DEFAULT_OOS_CONFIG),
        help="OOS period YAML. Default: configs/oos_period.yaml",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=["equal_weight", "sixty_forty", "risk_parity"],
        choices=["equal_weight", "sixty_forty", "risk_parity"],
        help="Which baselines to export turnover for.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=252,
        help="Lookback window (days) for risk_parity (inverse-vol placeholder).",
    )
    parser.add_argument(
        "--rebalance-freq",
        type=int,
        default=21,
        help="Rebalance frequency in trading days (used for drift segments). Default: 21 (~monthly).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_RESULTS / "oos_canonical" / "per_window"),
        help="Directory to write per-window turnover CSVs.",
    )
    args = parser.parse_args()

    # Resolve paths
    oos = load_oos_period(Path(args.oos_config))
    universe_list = load_universe(Path(args.universe))
    if args.returns:
        returns_path = Path(args.returns)
    else:
        try:
            returns_path = autodetect_returns(DEFAULT_RETURNS_CANDIDATES)
        except FileNotFoundError as e:
            print(str(e))
            sys.exit(1)

    print("=== EXPORT PER-WINDOW TURNOVER (BASELINES) ===")
    print(f"Returns: {returns_path}")
    print(f"OOS: {oos.start.date()} -> {oos.end.date()}")
    print(f"Universe file: {args.universe}")
    print(f"Strategies: {', '.join(args.strategies)}")
    print(f"Output dir: {args.output_dir}")
    print("=============================================")

    # Load returns and filter universe + OOS period
    rets_all = load_returns(returns_path)
    rets_oos = rets_all.loc[(rets_all.index >= oos.start) & (rets_all.index <= oos.end)].copy()
    rets_oos = filter_universe(rets_oos, universe_list)

    # Remove columns with all NaNs in OOS
    rets_oos = rets_oos.dropna(axis=1, how="all")
    # Fill sparse missing with 0 for drift segments (assume 0 return when missing)
    rets_oos = rets_oos.fillna(0.0)

    # Build rebalance schedule: first available business day of each month in OOS
    schedule = first_business_days(rets_oos.index, oos.start, oos.end)
    if len(schedule) < 2:
        print("Not enough rebalance points in OOS period to compute turnover.")
        sys.exit(0)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_stats = []

    # Strategy loop
    for strat in args.strategies:
        print(f"\n→ Processing strategy: {strat}")
        # Determine asset set for this strategy
        strat_cols = list(rets_oos.columns)

        # Initialize
        w_prev = pd.Series(0.0, index=strat_cols)
        last_date = schedule[0]
        turnovers: List[Tuple[pd.Timestamp, float]] = []

        # First rebalance at schedule[0]
        if strat == "equal_weight":
            w_tgt = target_weights_equal_weight(strat_cols)
        elif strat == "sixty_forty":
            w_tgt = target_weights_sixty_forty(strat_cols)
            # Expand to full index filling zeros (for turnover distance)
            w_tgt = w_tgt.reindex(strat_cols, fill_value=0.0)
        elif strat == "risk_parity":
            # Use lookback window ending at first rebalance date
            lb_end = schedule[0]
            lb_start = rets_oos.index[rets_oos.index.searchsorted(lb_end) - args.lookback] if len(rets_oos) >= args.lookback else rets_oos.index[0]
            rets_win = rets_oos.loc[lb_start:lb_end, strat_cols]
            w_tgt = target_weights_inverse_vol(rets_win)
        else:
            raise ValueError(f"Unknown strategy: {strat}")

        # Compute initial turnover vs zeros (interpreted como entrada no portfólio)
        t0_turnover = turnover_one_way(w_prev, w_tgt)
        turnovers.append((schedule[0], t0_turnover))
        w_prev = w_tgt.copy()

        # Subsequent rebalances
        for d_next in schedule[1:]:
            # Drift previous weights with returns between last_date (exclusive) and d_next (inclusive)
            seg = rets_oos.loc[(rets_oos.index > last_date) & (rets_oos.index <= d_next), strat_cols]
            w_pre = compute_pretrade_weights(w_prev, seg)

            # Build target weights at d_next
            if strat == "equal_weight":
                w_tgt = target_weights_equal_weight(strat_cols)
            elif strat == "sixty_forty":
                w_tgt = target_weights_sixty_forty(strat_cols).reindex(strat_cols, fill_value=0.0)
            elif strat == "risk_parity":
                # lookback ending at d_next
                end_pos = rets_oos.index.searchsorted(d_next)
                start_pos = max(0, end_pos - args.lookback)
                rets_win = rets_oos.iloc[start_pos:end_pos][strat_cols]
                w_tgt = target_weights_inverse_vol(rets_win)
            else:
                raise ValueError(f"Unknown strategy: {strat}")

            # Turnover and update
            tovr = turnover_one_way(w_pre, w_tgt)
            turnovers.append((d_next, tovr))
            w_prev = w_tgt.copy()
            last_date = d_next

        # Write per-window CSV
        df_out = pd.DataFrame(turnovers, columns=["date", "turnover"]).dropna()
        csv_path = out_dir / f"turnover_{strat}.csv"
        df_out.to_csv(csv_path, index=False)
        print(f"  ✓ Saved per-window turnover: {csv_path}")
        # Summary stats
        summary_stats.append({
            "strategy": strat,
            "turnover_median": float(df_out["turnover"].median()),
            "turnover_p95": float(df_out["turnover"].quantile(0.95)),
            "n_windows": int(len(df_out)),
        })

    # Save summary stats
    df_stats = pd.DataFrame(summary_stats)
    summary_path = out_dir.parent / "turnover_dist_stats.csv"
    df_stats.to_csv(summary_path, index=False)
    print(f"\n✓ Saved turnover distribution stats: {summary_path}")
    print(df_stats.to_string(index=False))


if __name__ == "__main__":
    main()
