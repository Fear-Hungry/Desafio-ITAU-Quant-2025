"""Command line interface for the project.

Provides a unified interface to run portfolio optimization, backtesting, and
various analysis scripts. Commands are organized into categories:
- Core: optimize, backtest, show-settings
- Examples: run-example
- Research: compare-baselines, compare-estimators, grid-search, test-skill, walkforward
- Production: production-deploy

All scripts have been reorganized from the root directory into scripts/ subfolders.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from itau_quant.backtesting import run_backtest
from itau_quant.config import Settings, configure_logging, get_settings
from itau_quant.optimization.solvers import run_optimizer

__all__ = ["build_parser", "main"]

# Get project root directory (3 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="itau_quant CLI")
    parser.add_argument(
        "--structured-logs",
        dest="structured_logs",
        action="store_true",
        help="força logs estruturados em JSON",
    )
    parser.add_argument(
        "--plain-logs",
        dest="structured_logs",
        action="store_false",
        help="força logs texto simples",
    )
    parser.set_defaults(structured_logs=None)

    subparsers = parser.add_subparsers(dest="command", required=True)

    show = subparsers.add_parser("show-settings", help="Exibe as Settings resolvidas")
    show.add_argument("--json", action="store_true", help="Formato JSON")

    opt = subparsers.add_parser("optimize", help="Executa (ou simula) o otimizador")
    opt.add_argument("--config", type=str, help="Arquivo de configuração YAML")
    opt.add_argument("--no-dry-run", action="store_false", dest="dry_run", help="Executa otimização real")
    opt.add_argument("--json", action="store_true", help="Mostra resultado em JSON")
    opt.set_defaults(dry_run=True)

    back = subparsers.add_parser("backtest", help="Executa (ou simula) o backtest")
    back.add_argument("--config", type=str, help="Arquivo de configuração YAML")
    back.add_argument("--no-dry-run", action="store_false", dest="dry_run", help="Executa backtest real")
    back.add_argument("--json", action="store_true", help="Mostra resultado em JSON")
    back.set_defaults(dry_run=True)

    # Examples category
    example = subparsers.add_parser("run-example", help="Executa portfolio de exemplo (ARARA)")
    example.add_argument(
        "variant",
        choices=["arara", "robust"],
        help="Variante: 'arara' (básico) ou 'robust' (robusto)"
    )

    # Research category
    baselines = subparsers.add_parser("compare-baselines", help="Compara estratégias baseline (1/N, MV, RP)")

    estimators = subparsers.add_parser("compare-estimators", help="Compara estimadores de μ e Σ")

    grid = subparsers.add_parser("grid-search", help="Grid search de hiperparâmetros (shrinkage)")

    skill = subparsers.add_parser("test-skill", help="Testa skill de forecast de μ")

    walkforward = subparsers.add_parser("walkforward", help="Backtest walk-forward com validação temporal")

    # Production category
    production = subparsers.add_parser("production-deploy", help="Deploy sistema de produção (ERC)")
    production.add_argument(
        "--version",
        choices=["v1", "v2"],
        default="v2",
        help="Versão: 'v1' (básico) ou 'v2' (calibrado, recomendado)"
    )

    return parser


def _configure_logging(structured: bool | None, settings: Settings, command: str) -> None:
    configure_logging(settings=settings, structured=structured, context={"command": command})


def _print_payload(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")


def _run_script(script_path: Path) -> int:
    """Execute a Python script and return its exit code."""
    if not script_path.exists():
        print(f"Error: Script not found at {script_path}", file=sys.stderr)
        return 1

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            check=False
        )
        return result.returncode
    except Exception as e:
        print(f"Error running script: {e}", file=sys.stderr)
        return 1


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    settings = get_settings()
    _configure_logging(args.structured_logs, settings, args.command)

    try:
        # Core commands
        if args.command == "show-settings":
            payload = settings.to_dict()
            _print_payload(payload, as_json=args.json)
        elif args.command == "optimize":
            opt_result = run_optimizer(args.config, dry_run=args.dry_run, settings=settings)
            payload = opt_result.to_dict(include_weights=args.json)
            _print_payload(payload, as_json=args.json)
        elif args.command == "backtest":
            result = run_backtest(args.config, dry_run=args.dry_run, settings=settings)
            payload = result.to_dict(include_timeseries=args.json)
            _print_payload(payload, as_json=args.json)

        # Example commands
        elif args.command == "run-example":
            script_map = {
                "arara": "scripts/examples/run_portfolio_arara.py",
                "robust": "scripts/examples/run_portfolio_arara_robust.py",
            }
            script_path = PROJECT_ROOT / script_map[args.variant]
            return _run_script(script_path)

        # Research commands
        elif args.command == "compare-baselines":
            script_path = PROJECT_ROOT / "scripts/research/run_baselines_comparison.py"
            return _run_script(script_path)
        elif args.command == "compare-estimators":
            script_path = PROJECT_ROOT / "scripts/research/run_estimator_comparison.py"
            return _run_script(script_path)
        elif args.command == "grid-search":
            script_path = PROJECT_ROOT / "scripts/research/run_grid_search_shrinkage.py"
            return _run_script(script_path)
        elif args.command == "test-skill":
            script_path = PROJECT_ROOT / "scripts/research/run_mu_skill_test.py"
            return _run_script(script_path)
        elif args.command == "walkforward":
            script_path = PROJECT_ROOT / "scripts/research/run_backtest_walkforward.py"
            return _run_script(script_path)

        # Production commands
        elif args.command == "production-deploy":
            script_map = {
                "v1": "scripts/production/run_portfolio_production_erc.py",
                "v2": "scripts/production/run_portfolio_production_erc_v2.py",
            }
            script_path = PROJECT_ROOT / script_map[args.version]
            return _run_script(script_path)

        else:  # pragma: no cover - defensive fallback
            parser.error(f"Unknown command: {args.command}")

    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
