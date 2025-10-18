"""Configuração padronizada de *logging* para o projeto."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Mapping

from .settings import Settings, get_settings

__all__ = ["JSONFormatter", "configure_logging"]


_RESERVED = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
}


class JSONFormatter(logging.Formatter):
    """Formatador que serializa ``LogRecord`` em JSON."""

    def __init__(self, *, default_context: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self._default_context = dict(default_context or {})

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - exercised via tests
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception details when present.
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = record.stack_info

        payload.update(self._default_context)

        for key, value in record.__dict__.items():
            if key in _RESERVED:
                continue
            payload.setdefault(key, value)

        return json.dumps(payload, ensure_ascii=False)


def _ensure_log_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def configure_logging(
    *,
    settings: Settings | None = None,
    level: int | str = logging.INFO,
    structured: bool | None = None,
    module_levels: Mapping[str, int | str] | None = None,
    stream: IO[str] | None = None,
    context: Mapping[str, Any] | None = None,
    log_file: Path | None = None,
) -> None:
    """Configure the root logger using the project defaults.

    Parameters
    ----------
    settings:
        Instância de :class:`Settings` a ser utilizada. Quando ``None`` o
        *singleton* de :func:`get_settings` é empregado.
    level:
        Nível padrão para o *root logger*.
    structured:
        Se ``True`` utiliza :class:`JSONFormatter`. Quando ``None`` utiliza o
        valor padrão definido em ``settings.structured_logging``.
    module_levels:
        Mapeamento ``logger -> level`` para ajustes finos.
    stream:
        Alvo para o ``StreamHandler`` principal; por padrão ``sys.stderr``.
    context:
        Campos extras aplicados a todos os registros (ex.: ``{"seed": 42}``).
    log_file:
        Caminho alternativo para escrever uma cópia dos logs (append). Caso
        ``None`` utiliza ``settings.logs_dir / 'itau_quant.log'``.
    """

    settings = settings or get_settings()
    structured = settings.structured_logging if structured is None else structured

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG)

    formatter: logging.Formatter
    if structured:
        formatter = JSONFormatter(default_context=context)
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    stream_handler = logging.StreamHandler(stream)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    file_target = log_file or (settings.logs_dir / "itau_quant.log")
    try:
        _ensure_log_dir(file_target.parent)
        file_handler = logging.FileHandler(file_target, encoding="utf-8")
    except OSError:  # pragma: no cover - filesystem issues exercised indirectly
        file_handler = None
    else:
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    if module_levels:
        for logger_name, logger_level in module_levels.items():
            logging.getLogger(logger_name).setLevel(logger_level)
