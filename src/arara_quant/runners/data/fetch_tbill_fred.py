#!/usr/bin/env python3
"""
Fetch daily 3M T-Bill yields from FRED and export a risk-free daily series.

Outputs
  data/processed/riskfree_tbill_daily.csv with columns:
    - date (YYYY-MM-DD)
    - rf_daily  (decimal daily return, approx: annual_rate / 252)
    - yield     (annualized decimal yield from FRED)

Usage
  poetry run python -m arara_quant.runners.data.fetch_tbill_fred --start 2010-01-01 --end 2025-12-31

Requires: pandas-datareader (already declared in pyproject).
Note: Network access required to reach FRED.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from pandas_datareader import data as web

from arara_quant.config import get_settings

DEFAULT_OUTPUT = get_settings().processed_data_dir / "riskfree_tbill_daily.csv"


def fetch_tbill(start: str, end: str) -> pd.DataFrame:
    # DGS3MO: 3-Month Treasury Constant Maturity Rate, Daily, Percent, Not Seasonally Adjusted
    df = web.DataReader("DGS3MO", "fred", start, end)
    df = df.rename(columns={"DGS3MO": "yield"})
    df.index = pd.to_datetime(df.index)
    df = df.reset_index().rename(columns={"DATE": "date", "index": "date"})
    # Convert percent to decimal if needed
    y = pd.to_numeric(df["yield"], errors="coerce")
    # DGS3MO is given in percent; convert to decimal
    y = y / 100.0
    df["yield"] = y
    # Approximate daily risk-free: annual_rate / 252
    df["rf_daily"] = df["yield"] / 252.0
    df = df.dropna(subset=["rf_daily"])  # drop days with missing yield
    return df[["date", "rf_daily", "yield"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch daily T-Bill from FRED")
    parser.add_argument("--start", type=str, default="2010-01-01")
    parser.add_argument("--end", type=str, default=datetime.today().strftime("%Y-%m-%d"))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching DGS3MO from FRED: {args.start} to {args.end}")
    df = fetch_tbill(args.start, args.end)
    df.to_csv(out_path, index=False)
    print(f"✓ Saved: {out_path}  (rows={len(df)})")


if __name__ == "__main__":
    main()
