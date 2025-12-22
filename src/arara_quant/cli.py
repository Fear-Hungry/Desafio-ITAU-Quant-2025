"""Command line interface for the project.

Provides a unified interface to run portfolio optimization, backtesting, and
various analysis runners. Commands are organized into categories:
- Core: optimize, backtest, show-settings
- Examples: run-example
- Research: compare-baselines, compare-estimators, grid-search, test-skill, walkforward
- Production: production-deploy

Runner modules live under ``arara_quant.runners`` for in-process execution.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from arara_quant.config import Settings, configure_logging, get_settings
from arara_quant.utils.yaml_loader import load_yaml_text

__all__ = ["build_parser", "main"]

if TYPE_CHECKING:  # pragma: no cover
    from arara_quant.backtesting import BacktestResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="arara_quant CLI")
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
    opt.add_argument(
        "--metaheuristic-config",
        type=str,
        help="Arquivo YAML com configuração de meta-heurística (GA)",
    )
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
        "--output-dir",
        default="outputs/reports",
        help="Diretório para salvar resultados",
    )
    pipeline.add_argument("--json", action="store_true", help="Output em formato JSON")

    data = subparsers.add_parser(
        "data",
        help="Baixa (ou reutiliza cache) e prepara dados",
    )
    data.add_argument("--start", help="Data inicial (YYYY-MM-DD)")
    data.add_argument("--end", help="Data final (YYYY-MM-DD)")
    data.add_argument(
        "--raw-file",
        default="prices_arara.csv",
        help="CSV de preços salvo em data/raw/ (compatibilidade).",
    )
    data.add_argument(
        "--processed-file",
        default="returns_arara.parquet",
        help="Parquet de retornos salvo em data/processed/ (compatibilidade).",
    )
    data.add_argument(
        "--force-download",
        action="store_true",
        help="Ignora cache e força download (requer rede).",
    )
    data.add_argument("--json", action="store_true", help="Output em JSON")

    estimate = subparsers.add_parser(
        "estimate",
        help="Estima μ/Σ a partir de returns",
    )
    estimate.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config YAML (usa estimators.{mu,sigma} quando presente).",
    )
    estimate.add_argument(
        "--returns-file",
        default=None,
        help="Arquivo Parquet/CSV com returns (path absoluto ou relativo).",
    )
    estimate.add_argument(
        "--window",
        type=int,
        default=None,
        help="Janela de estimação em dias (default: usa YAML ou 252).",
    )
    estimate.add_argument(
        "--mu-method",
        default=None,
        help="Estimador de μ (simple, huber, shrunk_50, student_t).",
    )
    estimate.add_argument(
        "--cov-method",
        default=None,
        help="Estimador de Σ (ledoit_wolf, nonlinear, oas, mincovdet, sample).",
    )
    estimate.add_argument(
        "--huber-delta",
        type=float,
        default=None,
        help="Delta do Huber (quando mu-method=huber).",
    )
    estimate.add_argument(
        "--shrink-strength",
        type=float,
        default=None,
        help="Strength do shrinkage (quando mu-method=shrunk_50).",
    )
    estimate.add_argument(
        "--student-t-nu",
        type=float,
        default=None,
        help="Graus de liberdade (quando mu-method=student_t).",
    )
    annualize = estimate.add_mutually_exclusive_group()
    annualize.add_argument(
        "--annualize",
        dest="annualize",
        action="store_true",
        help="Anualiza μ/Σ (default).",
    )
    annualize.add_argument(
        "--no-annualize",
        dest="annualize",
        action="store_false",
        help="Não anualiza μ/Σ (mantém unidades diárias).",
    )
    estimate.set_defaults(annualize=True)
    estimate.add_argument(
        "--mu-output",
        default="mu_estimate.parquet",
        help="Nome do output Parquet para μ (em data/processed/).",
    )
    estimate.add_argument(
        "--cov-output",
        default="cov_estimate.parquet",
        help="Nome do output Parquet para Σ (em data/processed/).",
    )
    estimate.add_argument("--json", action="store_true", help="Output em JSON")

    validate_configs = subparsers.add_parser(
        "validate-configs", help="Valida arquivos YAML em configs/"
    )
    validate_configs.add_argument("--json", action="store_true", help="Output em JSON")

    update_turnover = subparsers.add_parser(
        "update-readme-turnover",
        help="Atualiza a tabela de turnover no README.md",
    )
    update_turnover.add_argument(
        "--readme",
        type=Path,
        default=None,
        help="Path to README.md (default: <project_root>/README.md).",
    )
    update_turnover.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Baseline turnover summary CSV (default: outputs/results/oos_canonical/turnover_dist_stats.csv).",
    )
    update_turnover.add_argument(
        "--per-window-prism",
        type=Path,
        default=None,
        help="PRISM-R per-window CSV (default: outputs/reports/walkforward/per_window_results.csv).",
    )
    update_turnover.add_argument(
        "--prism-trades",
        type=Path,
        default=None,
        help="PRISM-R trade-level turnover CSV (default: outputs/reports/walkforward/trades.csv).",
    )
    update_turnover.add_argument(
        "--oos-config",
        type=Path,
        default=None,
        help="OOS period YAML override (default: configs/oos_period.yaml).",
    )
    update_turnover.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Overwrite existing values (not only placeholders).",
    )

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


def _load_metaheuristic_override(path_str: str, settings: Settings) -> dict[str, Any]:
    candidate = Path(path_str)
    if not candidate.is_absolute():
        candidate = (settings.project_root / candidate).resolve()
    if not candidate.exists():
        raise FileNotFoundError(
            f"Metaheuristic config not found: {candidate}"
        )
    return load_yaml_text(candidate.read_text(encoding="utf-8"))


def _print_payload(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")


def _resolve_config_relative(
    value: str | Path,
    *,
    base: Path,
    settings: Settings,
) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (base / candidate).resolve()
    if not candidate.exists():
        fallback = (settings.project_root / Path(value)).resolve()
        if fallback.exists():
            candidate = fallback
    return candidate


def _infer_returns_from_config(
    config_path: str | None, *, settings: Settings
) -> str | None:
    if not config_path:
        return None

    config_file = Path(config_path)
    if not config_file.is_absolute():
        in_configs = settings.configs_dir / config_file
        config_file = (
            in_configs if in_configs.exists() else settings.project_root / config_file
        )
    config_file = config_file.expanduser().resolve()
    if not config_file.exists():
        return None

    raw = load_yaml_text(config_file.read_text(encoding="utf-8"))
    data_section = raw.get("data", {})
    if not isinstance(data_section, Mapping):
        return None

    returns_entry = data_section.get("returns") or data_section.get("returns_path")
    if not returns_entry:
        return None

    returns_path = _resolve_config_relative(
        str(returns_entry), base=config_file.parent, settings=settings
    )
    return str(returns_path)


def _run_runner_module(module: str, argv: list[str] | None = None) -> int:
    """Execute a runner module in-process and return its exit code."""
    saved_argv = sys.argv
    sys.argv = [module]
    if argv:
        sys.argv.extend(argv)
    try:
        runpy.run_module(module, run_name="__main__")
        return 0
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        print(code, file=sys.stderr)
        return 1
    finally:
        sys.argv = saved_argv


def _generate_wf_report(
    result: BacktestResult, output_dir: str = "outputs/reports/walkforward"
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

    from arara_quant.evaluation.plots.walkforward import plot_walkforward_summary
    from arara_quant.evaluation.walkforward_report import (
        build_per_window_table,
        format_wf_summary_markdown,
        identify_stress_periods,
    )

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

    # 2. Export raw per-window metrics for downstream processing
    raw_per_window = output_path / "per_window_results_raw.csv"
    result.split_metrics.to_csv(raw_per_window, index=False)
    print(f"✓ Raw per-window metrics: {raw_per_window}")

    # 3. Export formatted per-window results table
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

    # 4. Persist trade-level diagnostics for turnover/cost analysis
    if result.trades is not None and not result.trades.empty:
        trades_path = output_path / "trades.csv"
        result.trades.to_csv(trades_path, index=False)
        print(f"✓ Trade diagnostics: {trades_path}")

    if result.weights is not None and not result.weights.empty:
        weights_path = output_path / "weights_history.csv"
        weights_export = result.weights.reset_index()
        weights_export.rename(columns={"index": "date"}, inplace=True)
        weights_export.to_csv(weights_path, index=False)
        print(f"✓ Weights history: {weights_path}")

    # 5. Identify and export stress periods
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

    # 6. Generate visualizations
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
    print("  - per_window_results_raw.csv")
    print("  - per_window_results.csv")
    print("  - per_window_results.md")
    if stress_periods:
        print("  - stress_periods.md")
    if result.trades is not None and not result.trades.empty:
        print("  - trades.csv")
    if result.weights is not None and not result.weights.empty:
        print("  - weights_history.csv")
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
            from arara_quant.optimization.solvers import run_optimizer

            meta_override = None
            if getattr(args, "metaheuristic_config", None):
                meta_override = _load_metaheuristic_override(
                    args.metaheuristic_config, settings
                )
            opt_result = run_optimizer(
                args.config,
                dry_run=args.dry_run,
                settings=settings,
                metaheuristic_override=meta_override,
            )
            payload = opt_result.to_dict(include_weights=args.json)
            _print_payload(payload, as_json=args.json)
        elif args.command == "backtest":
            from arara_quant.backtesting import run_backtest

            result = run_backtest(args.config, dry_run=args.dry_run, settings=settings)

            # Generate walk-forward report if requested
            if hasattr(args, "wf_report") and args.wf_report and not args.dry_run:
                _generate_wf_report(result, output_dir="outputs/reports/walkforward")

            payload = result.to_dict(include_timeseries=args.json)
            _print_payload(payload, as_json=args.json)

        # Pipeline orchestration
        elif args.command == "run-full-pipeline":
            from arara_quant.pipeline import run_full_pipeline

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

        elif args.command == "data":
            from arara_quant.pipeline.data import download_and_prepare_data

            result = download_and_prepare_data(
                start=args.start,
                end=args.end,
                raw_file_name=args.raw_file,
                processed_file_name=args.processed_file,
                force_download=args.force_download,
                settings=settings,
            )
            _print_payload(result, as_json=args.json)

        elif args.command == "estimate":
            from arara_quant.pipeline.estimation import estimate_parameters

            returns_from_yaml = _infer_returns_from_config(args.config, settings=settings)
            returns_file = (
                args.returns_file or returns_from_yaml or "returns_arara.parquet"
            )

            result = estimate_parameters(
                returns_file=returns_file,
                window=args.window,
                mu_method=args.mu_method,
                cov_method=args.cov_method,
                huber_delta=args.huber_delta,
                shrink_strength=args.shrink_strength,
                student_t_nu=args.student_t_nu,
                annualize=args.annualize,
                mu_output=args.mu_output,
                cov_output=args.cov_output,
                config_path=args.config,
                settings=settings,
            )
            _print_payload(result, as_json=args.json)

        elif args.command == "validate-configs":
            from arara_quant.reports.validators import validate_configs

            result = validate_configs(settings)
            if args.json:
                payload = {
                    "validated": [str(path) for path in result.validated],
                    "errors": [
                        {"path": str(path), "error": message}
                        for path, message in result.errors
                    ],
                }
                print(json.dumps(payload, indent=2, sort_keys=True))
                return 1 if result.errors else 0

            errors_by_path = {path: message for path, message in result.errors}
            for config_file in result.validated:
                if config_file in errors_by_path:
                    print(
                        f"✗ {config_file.name}: {errors_by_path[config_file]}",
                        file=sys.stderr,
                    )
                else:
                    print(f"✓ {config_file.name}")
            return 1 if result.errors else 0

        elif args.command == "update-readme-turnover":
            from arara_quant.reports.generators import update_readme_turnover_stats

            updated = update_readme_turnover_stats(
                settings=settings,
                readme_path=args.readme,
                baseline_summary_csv=args.summary,
                prism_per_window_csv=args.per_window_prism,
                prism_trades_csv=args.prism_trades,
                oos_config_path=args.oos_config,
                force_overwrite=args.force_overwrite,
            )
            print(f"Updated rows: {updated}")
            return 0

        # Example commands
        elif args.command == "run-example":
            module_map = {
                "arara": "arara_quant.runners.examples.run_portfolio_arara",
                "robust": "arara_quant.runners.examples.run_portfolio_arara_robust",
            }
            return _run_runner_module(module_map[args.variant])

        # Research commands
        elif args.command == "compare-baselines":
            return _run_runner_module(
                "arara_quant.runners.research.run_baselines_comparison"
            )
        elif args.command == "compare-estimators":
            return _run_runner_module(
                "arara_quant.runners.research.run_estimator_comparison"
            )
        elif args.command == "grid-search":
            return _run_runner_module(
                "arara_quant.runners.research.run_grid_search_shrinkage"
            )
        elif args.command == "test-skill":
            return _run_runner_module("arara_quant.runners.research.run_mu_skill_test")
        elif args.command == "walkforward":
            return _run_runner_module(
                "arara_quant.runners.research.run_backtest_walkforward"
            )

        # Production commands
        elif args.command == "production-deploy":
            module_map = {
                "v1": "arara_quant.runners.production.run_portfolio_production_erc",
                "v2": "arara_quant.runners.production.run_portfolio_production_erc_v2",
            }
            return _run_runner_module(module_map[args.version])

        else:  # pragma: no cover - defensive fallback
            parser.error(f"Unknown command: {args.command}")

    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
