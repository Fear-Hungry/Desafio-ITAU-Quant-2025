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

from itau_quant.backtesting import BacktestResult, run_backtest
from itau_quant.config import Settings, configure_logging, get_settings
from itau_quant.evaluation.plots.walkforward import plot_walkforward_summary
from itau_quant.evaluation.walkforward_report import (
    build_per_window_table,
    format_wf_summary_markdown,
    identify_stress_periods,
)
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
    opt.add_argument(
        "--no-dry-run",
        action="store_false",
        dest="dry_run",
        help="Executa otimização real",
    )
    opt.add_argument("--json", action="store_true", help="Mostra resultado em JSON")
    opt.set_defaults(dry_run=True)

    back = subparsers.add_parser("backtest", help="Executa (ou simula) o backtest")
    back.add_argument("--config", type=str, help="Arquivo de configuração YAML")
    back.add_argument(
        "--no-dry-run",
        action="store_false",
        dest="dry_run",
        help="Executa backtest real",
    )
    back.add_argument("--json", action="store_true", help="Mostra resultado em JSON")
    back.add_argument(
        "--wf-report",
        action="store_true",
        help="Gera relatório walk-forward completo (figuras + tabelas)",
    )
    back.set_defaults(dry_run=True)

    # Pipeline orchestration
    pipeline = subparsers.add_parser(
        "run-full-pipeline",
        help="Executa pipeline completo: dados → estimação → otimização → backtest",
    )
    pipeline.add_argument(
        "--config", required=True, help="Arquivo YAML de configuração"
    )
    pipeline.add_argument("--start", help="Data inicial (YYYY-MM-DD)")
    pipeline.add_argument("--end", help="Data final (YYYY-MM-DD)")
    pipeline.add_argument(
        "--skip-download",
        action="store_true",
        help="Usa dados cached (mais rápido para testes)",
    )
    pipeline.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Não executa backtest (apenas otimização)",
    )
    pipeline.add_argument(
        "--output-dir", default="reports", help="Diretório para salvar resultados"
    )
    pipeline.add_argument("--json", action="store_true", help="Output em formato JSON")

    # Examples category
    example = subparsers.add_parser(
        "run-example", help="Executa portfolio de exemplo (ARARA)"
    )
    example.add_argument(
        "variant",
        choices=["arara", "robust"],
        help="Variante: 'arara' (básico) ou 'robust' (robusto)",
    )

    # Research category
    subparsers.add_parser(
        "compare-baselines", help="Compara estratégias baseline (1/N, MV, RP)"
    )

    subparsers.add_parser("compare-estimators", help="Compara estimadores de μ e Σ")

    subparsers.add_parser(
        "grid-search", help="Grid search de hiperparâmetros (shrinkage)"
    )

    subparsers.add_parser("test-skill", help="Testa skill de forecast de μ")

    subparsers.add_parser(
        "walkforward", help="Backtest walk-forward com validação temporal"
    )

    # Production category
    production = subparsers.add_parser(
        "production-deploy", help="Deploy sistema de produção (ERC)"
    )
    production.add_argument(
        "--version",
        choices=["v1", "v2"],
        default="v2",
        help="Versão: 'v1' (básico) ou 'v2' (calibrado, recomendado)",
    )

    return parser


def _configure_logging(
    structured: bool | None, settings: Settings, command: str
) -> None:
    configure_logging(
        settings=settings, structured=structured, context={"command": command}
    )


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
            [sys.executable, str(script_path)], cwd=PROJECT_ROOT, check=False
        )
        return result.returncode
    except Exception as e:
        print(f"Error running script: {e}", file=sys.stderr)
        return 1


def _generate_wf_report(
    result: BacktestResult, output_dir: str = "reports/walkforward"
) -> None:
    """Generate comprehensive walk-forward report with visualizations and tables.

    Parameters
    ----------
    result : BacktestResult
        Backtest result containing split_metrics
    output_dir : str
        Directory to save report files
    """
    if result.split_metrics is None or result.split_metrics.empty:
        print(
            "No split_metrics available. Skipping walk-forward report.", file=sys.stderr
        )
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n=== Generating Walk-Forward Report ===")

    # 1. Generate summary statistics markdown
    if result.walkforward_summary is not None:
        summary_md = format_wf_summary_markdown(result.walkforward_summary)
        summary_file = output_path / "summary_stats.md"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(summary_md)
        print(f"✓ Summary statistics: {summary_file}")

    # 2. Export per-window results table
    per_window_table_csv = build_per_window_table(result.split_metrics, format="csv")
    csv_file = output_path / "per_window_results.csv"
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write(per_window_table_csv)
    print(f"✓ Per-window results: {csv_file}")

    # Also save markdown version
    per_window_table_md = build_per_window_table(
        result.split_metrics, format="markdown"
    )
    md_file = output_path / "per_window_results.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(per_window_table_md)
    print(f"✓ Per-window results (markdown): {md_file}")

    # 3. Identify and export stress periods
    stress_periods = identify_stress_periods(result.split_metrics)
    if stress_periods:
        stress_lines = ["# Identified Stress Periods\n"]
        for period in stress_periods:
            stress_lines.append(
                f"- **{period.label}** ({period.test_start} to {period.test_end}): "
                f"Sharpe={period.sharpe:.2f}, Drawdown={period.max_drawdown:.2%}, "
                f"Return={period.return_:.2%}\n"
            )
        stress_file = output_path / "stress_periods.md"
        with open(stress_file, "w", encoding="utf-8") as f:
            f.writelines(stress_lines)
        print(f"✓ Stress periods: {stress_file}")

    # 4. Generate visualizations
    try:
        fig = plot_walkforward_summary(result.split_metrics)
        fig_file = output_path / "walkforward_analysis.png"
        fig.savefig(fig_file, dpi=150, bbox_inches="tight")
        print(f"✓ Walk-forward figures: {fig_file}")
    except Exception as e:
        print(f"⚠ Failed to generate visualizations: {e}", file=sys.stderr)

    print(f"\nWalk-forward report saved to: {output_path}/")
    print("Files generated:")
    print("  - summary_stats.md")
    print("  - per_window_results.csv")
    print("  - per_window_results.md")
    if stress_periods:
        print("  - stress_periods.md")
    print("  - walkforward_analysis.png")
    print()


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
            opt_result = run_optimizer(
                args.config, dry_run=args.dry_run, settings=settings
            )
            payload = opt_result.to_dict(include_weights=args.json)
            _print_payload(payload, as_json=args.json)
        elif args.command == "backtest":
            result = run_backtest(args.config, dry_run=args.dry_run, settings=settings)

            # Generate walk-forward report if requested
            if hasattr(args, "wf_report") and args.wf_report and not args.dry_run:
                _generate_wf_report(result, output_dir="reports/walkforward")

            payload = result.to_dict(include_timeseries=args.json)
            _print_payload(payload, as_json=args.json)

        # Pipeline orchestration
        elif args.command == "run-full-pipeline":
            from itau_quant.pipeline import run_full_pipeline

            result = run_full_pipeline(
                config_path=args.config,
                start=args.start,
                end=args.end,
                skip_download=args.skip_download,
                skip_backtest=args.skip_backtest,
                output_dir=args.output_dir,
                settings=settings,
            )
            _print_payload(result, as_json=args.json)

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
