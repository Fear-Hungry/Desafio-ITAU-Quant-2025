from __future__ import annotations

import io
import json
import logging
from pathlib import Path

from arara_quant.config.logging_conf import configure_logging
from arara_quant.config.settings import Settings, reset_settings_cache


def _cleanup_logging() -> None:
    root = logging.getLogger()
    for handler in root.handlers[:]:
        handler.close()
        root.removeHandler(handler)


def teardown_function() -> None:  # pragma: no cover - cleanup helper
    _cleanup_logging()
    reset_settings_cache()


def test_configure_logging_structured_output(tmp_path: Path) -> None:
    stream = io.StringIO()
    settings = Settings.from_env(
        overrides={
            "project_root": tmp_path,
            "LOGS_DIR": tmp_path / "logs",
        }
    )

    configure_logging(
        settings=settings, structured=True, stream=stream, context={"run_id": "unit"}
    )

    logger = logging.getLogger("config.tests")
    logger.info("structured message", extra={"step": "load"})

    raw_line = stream.getvalue().strip().splitlines()[-1]
    payload = json.loads(raw_line)
    assert payload["run_id"] == "unit"
    assert payload["step"] == "load"
    assert payload["logger"] == "config.tests"

    log_file = settings.logs_dir / "arara_quant.log"
    assert log_file.exists()


def test_configure_logging_plaintext(tmp_path: Path) -> None:
    stream = io.StringIO()
    settings = Settings.from_env(
        overrides={"project_root": tmp_path, "LOGS_DIR": tmp_path / "logs"}
    )

    configure_logging(settings=settings, structured=False, stream=stream)

    logger = logging.getLogger("config.tests")
    logger.warning("plain message")

    output = stream.getvalue()
    assert "plain message" in output
    assert "WARNING" in output
