"""Command line interface for the project.

Os motores de otimização e backtesting ainda estão sendo construídos, mas o
CLI já oferece comandos utilitários que centralizam a resolução de
configurações e inicialização de *logging*. Isso facilita a automação desde
cedo sem duplicar lógica em *scripts* externos.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Iterable

from itau_quant.backtesting import run_backtest
from itau_quant.config import Settings, configure_logging, get_settings
from itau_quant.optimization.solvers import run_optimizer

__all__ = ["build_parser", "main"]


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

    return parser


def _configure_logging(structured: bool | None, settings: Settings, command: str) -> None:
    configure_logging(settings=settings, structured=structured, context={"command": command})


def _print_payload(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    settings = get_settings()
    _configure_logging(args.structured_logs, settings, args.command)

    try:
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
        else:  # pragma: no cover - defensive fallback
            parser.error(f"Unknown command: {args.command}")
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
