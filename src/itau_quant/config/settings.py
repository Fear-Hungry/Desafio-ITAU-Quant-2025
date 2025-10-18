"""Carrega configurações globais para o projeto.

As funções expostas aqui fornecem um *singleton* barato de configurações com
fallback sensato para ambientes de desenvolvimento. A implementação utiliza
apenas a biblioteca padrão, mas oferece uma experiência semelhante ao
``BaseSettings`` do Pydantic, incluindo suporte a arquivos ``.env`` e parsing
automático de tipos primitivos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Any, Mapping, MutableMapping

from .constants import DEFAULT_BASE_CURRENCY

__all__ = [
    "ENV_PREFIX",
    "Settings",
    "get_settings",
    "load_env_file",
    "reset_settings_cache",
]


ENV_PREFIX = "ITAU_QUANT_"
"""Prefixo utilizado para todas as variáveis de ambiente do projeto."""

_BOOL_TRUE = {"1", "true", "yes", "on"}
_BOOL_FALSE = {"0", "false", "no", "off"}


def _coerce_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in _BOOL_TRUE:
        return True
    if lowered in _BOOL_FALSE:
        return False
    raise ValueError(f"Cannot interpret '{value}' as boolean")


def _parse_value(value: str, *, target: type) -> Any:
    if target is bool:
        return _coerce_bool(value)
    if target is int:
        return int(value)
    if target is float:
        return float(value)
    return value


def _expand_path(path_str: str, *, base: Path) -> Path:
    candidate = Path(path_str).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate


def load_env_file(path: Path) -> Mapping[str, str]:
    """Parse a ``.env`` style file returning a mapping of key/values.

    Blank lines and comments starting with ``#`` are ignored. Only the first
    ``=`` in each line is treated as the separator.
    """

    entries: MutableMapping[str, str] = {}
    if not path.exists():
        return entries

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        entries[key.strip()] = value.strip()
    return entries


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(slots=True, frozen=True)
class Settings:
    """Conjunto imutável de configurações globais.

    A classe guarda caminhos utilizados pelo pipeline (dados, relatórios),
    *flags* de execução e metadados básicos como ``environment`` e
    ``base_currency``. A construção padrão utiliza pastas relativas ao root do
    repositório, mas todas as chaves podem ser sobrescritas via variáveis de
    ambiente prefixadas com :data:`ENV_PREFIX` ou via argumentos explícitos.
    """

    project_root: Path
    data_dir: Path
    raw_data_dir: Path
    processed_data_dir: Path
    notebooks_dir: Path
    reports_dir: Path
    configs_dir: Path
    logs_dir: Path
    cache_dir: Path
    environment: str
    random_seed: int
    base_currency: str
    structured_logging: bool

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_root": str(self.project_root),
            "data_dir": str(self.data_dir),
            "raw_data_dir": str(self.raw_data_dir),
            "processed_data_dir": str(self.processed_data_dir),
            "notebooks_dir": str(self.notebooks_dir),
            "reports_dir": str(self.reports_dir),
            "configs_dir": str(self.configs_dir),
            "logs_dir": str(self.logs_dir),
            "cache_dir": str(self.cache_dir),
            "environment": self.environment,
            "random_seed": self.random_seed,
            "base_currency": self.base_currency,
            "structured_logging": self.structured_logging,
        }

    @classmethod
    def from_env(
        cls,
        *,
        overrides: Mapping[str, Any] | None = None,
        env_file: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "Settings":
        """Create :class:`Settings` merging defaults, env file, env vars and overrides."""

        overrides = dict(overrides or {})

        env_mapping: MutableMapping[str, str] = {}
        if env_file is not None:
            explicit_env_path = Path(env_file).expanduser()
            env_mapping.update(load_env_file(explicit_env_path))

        system_environ = dict(environ or os.environ)

        project_root_value = overrides.pop("project_root", None)
        if project_root_value is None:
            project_root_value = env_mapping.get(f"{ENV_PREFIX}PROJECT_ROOT")
        if project_root_value is None:
            project_root_value = system_environ.get(f"{ENV_PREFIX}PROJECT_ROOT")

        if project_root_value is None:
            project_root = _project_root()
        else:
            project_root = Path(str(project_root_value)).expanduser().resolve()

        default_env_path = project_root / ".env"
        if env_file is None and default_env_path.exists():
            env_mapping.update(load_env_file(default_env_path))

        # System environment has the highest precedence.
        env_mapping.update(system_environ)

        def pull(name: str, *, default: Any, target: type) -> Any:
            key = f"{ENV_PREFIX}{name}"
            if name in overrides:
                value = overrides.pop(name)
                if target is Path:
                    return _expand_path(str(value), base=project_root)
                if isinstance(value, str) and target is bool:
                    return _coerce_bool(value)
                if isinstance(value, str) and target in {int, float}:
                    return _parse_value(value, target=target)
                return value
            if key in env_mapping:
                raw = env_mapping[key]
                if target is Path:
                    return _expand_path(raw, base=project_root)
                return _parse_value(raw, target=target)
            if target is Path:
                return _expand_path(str(default), base=project_root)
            return default

        data_dir = pull("DATA_DIR", default="data", target=Path)
        raw_data_dir = pull("RAW_DATA_DIR", default=data_dir / "raw", target=Path)
        processed_data_dir = pull("PROCESSED_DATA_DIR", default=data_dir / "processed", target=Path)
        notebooks_dir = pull("NOTEBOOKS_DIR", default="notebooks", target=Path)
        reports_dir = pull("REPORTS_DIR", default="reports", target=Path)
        configs_dir = pull("CONFIGS_DIR", default="configs", target=Path)
        logs_dir = pull("LOGS_DIR", default="logs", target=Path)
        cache_dir = pull("CACHE_DIR", default=".cache", target=Path)

        environment = pull("ENVIRONMENT", default="development", target=str)
        random_seed = pull("RANDOM_SEED", default=42, target=int)
        base_currency = pull("BASE_CURRENCY", default=DEFAULT_BASE_CURRENCY, target=str)
        structured_logging = pull("STRUCTURED_LOGGING", default=True, target=bool)

        if overrides:
            unknown = ", ".join(sorted(overrides))
            raise KeyError(f"Unknown override(s): {unknown}")

        return cls(
            project_root=project_root,
            data_dir=data_dir,
            raw_data_dir=raw_data_dir,
            processed_data_dir=processed_data_dir,
            notebooks_dir=notebooks_dir,
            reports_dir=reports_dir,
            configs_dir=configs_dir,
            logs_dir=logs_dir,
            cache_dir=cache_dir,
            environment=str(environment),
            random_seed=int(random_seed),
            base_currency=str(base_currency),
            structured_logging=bool(structured_logging),
        )


_SETTINGS_CACHE: Settings | None = None


def get_settings(**kwargs: Any) -> Settings:
    """Return cached settings, building them on the first call.

    When keyword arguments are provided the cache is bypassed and freshly
    computed settings are returned. The cache itself can be cleared with
    :func:`reset_settings_cache`.
    """

    global _SETTINGS_CACHE
    if kwargs:
        return Settings.from_env(**kwargs)
    if _SETTINGS_CACHE is None:
        _SETTINGS_CACHE = Settings.from_env()
    return _SETTINGS_CACHE


def reset_settings_cache() -> None:
    """Clear the singleton cache (useful for tests)."""

    global _SETTINGS_CACHE
    _SETTINGS_CACHE = None
