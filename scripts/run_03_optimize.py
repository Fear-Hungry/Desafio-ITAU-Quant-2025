#!/usr/bin/env python
"""Solve the portfolio optimization step using previously estimated μ/Σ.

This script is now a thin wrapper around the reusable pipeline.optimization module.
"""

from __future__ import annotations

import argparse

from arara_quant.config import Settings
from arara_quant.pipeline.optimization import optimize_portfolio
from arara_quant.utils.logging_config import get_logger, log_dict

logger = get_logger("scripts.run_03_optimize")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize the portfolio weights.")
    parser.add_argument(
        "--mu-file",
        default="mu_estimate.parquet",
        help="Input file under data/processed/ containing annualized μ.",
    )
    parser.add_argument(
        "--cov-file",
        default="cov_estimate.parquet",
        help="Input file under data/processed/ containing annualized Σ.",
    )
    parser.add_argument(
        "--risk-aversion",
        type=float,
        default=4.0,
        help="Risk-aversion coefficient λ for the mean-variance program.",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=0.15,
        help="Upper bound applied to each asset weight.",
    )
    parser.add_argument(
        "--turnover-cap",
        type=float,
        default=None,
        help="Optional ℓ₁ turnover cap (set to 0.10 for 10% limit, for example).",
    )
    parser.add_argument(
        "--output",
        default="optimized_weights.parquet",
        help="Parquet file under outputs/results/ storing optimized weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings.from_env()

    # Call the reusable module function
    try:
        result = optimize_portfolio(
            mu_file=args.mu_file,
            cov_file=args.cov_file,
            risk_aversion=args.risk_aversion,
            max_weight=args.max_weight,
            turnover_cap=args.turnover_cap,
            output_file=args.output,
            settings=settings,
        )
    except Exception as exc:
        logger.error("Failed to execute portfolio optimization: %s", exc)
        raise SystemExit(1) from exc

    log_dict(logger, "Optimization completed", result)


if __name__ == "__main__":
    main()
