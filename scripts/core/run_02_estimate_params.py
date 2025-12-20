#!/usr/bin/env python
"""Estimate expected returns (μ) and covariance (Σ) from processed returns.

This script is now a thin wrapper around the reusable pipeline.estimation module.
"""

from __future__ import annotations

import argparse

from arara_quant.config import Settings
from arara_quant.pipeline.estimation import estimate_parameters
from arara_quant.utils.logging_config import get_logger, log_dict

logger = get_logger("scripts.run_02_estimate_params")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate μ/Σ from historical returns."
    )
    parser.add_argument(
        "--returns-file",
        default="returns_arara.parquet",
        help="Input parquet file under data/processed/ with log returns.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=252,
        help="Number of most recent observations to use.",
    )
    parser.add_argument(
        "--mu-method",
        choices=("simple", "huber", "shrunk_50"),
        default="shrunk_50",
        help="Expected return estimator to apply.",
    )
    parser.add_argument(
        "--cov-method",
        choices=("ledoit_wolf", "oas", "mincovdet", "sample"),
        default="ledoit_wolf",
        help="Covariance estimator to apply (Ledoit-Wolf, OAS, MinCovDet ou amostral).",
    )
    parser.add_argument(
        "--huber-delta",
        type=float,
        default=1.5,
        help="Robustness threshold for the Huber estimator.",
    )
    annualize = parser.add_mutually_exclusive_group()
    annualize.add_argument(
        "--annualize",
        dest="annualize",
        action="store_true",
        help="Annualize μ/Σ assuming 252 trading days (default).",
    )
    annualize.add_argument(
        "--no-annualize",
        dest="annualize",
        action="store_false",
        help="Keep μ/Σ in daily units.",
    )
    parser.set_defaults(annualize=True)
    parser.add_argument(
        "--shrink-strength",
        type=float,
        default=0.5,
        help="Shrinkage intensity towards the prior when using shrunk_50.",
    )
    parser.add_argument(
        "--mu-output",
        default="mu_estimate.parquet",
        help="Output file under data/processed/ storing annualized μ.",
    )
    parser.add_argument(
        "--cov-output",
        default="cov_estimate.parquet",
        help="Output file under data/processed/ storing annualized Σ.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings.from_env()

    # Call the reusable module function
    try:
        result = estimate_parameters(
            returns_file=args.returns_file,
            window=args.window,
            mu_method=args.mu_method,
            cov_method=args.cov_method,
            huber_delta=args.huber_delta,
            annualize=args.annualize,
            mu_output=args.mu_output,
            cov_output=args.cov_output,
            shrink_strength=args.shrink_strength,
            settings=settings,
        )
    except Exception as exc:
        logger.error("Failed to execute parameter estimation: %s", exc)
        raise SystemExit(1) from exc

    log_dict(logger, "Parameter estimation completed", result)


if __name__ == "__main__":
    main()
