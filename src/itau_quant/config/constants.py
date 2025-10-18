"""Constantes centrais utilizadas em múltiplos módulos.

O arquivo consolida valores numéricos (dias úteis, conversões de basis
points), *strings* recorrentes (nomes de colunas de dados) e pequenos
utilitários de apoio. A centralização evita a propagação de literais mágicos
espalhados pelo projeto.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "BPS",
    "BPS_TO_PCT",
    "DEFAULT_BASE_CURRENCY",
    "DEFAULT_RISK_FREE_RATE",
    "COLUMN_ADV",
    "COLUMN_DATE",
    "COLUMN_PRICE",
    "COLUMN_RETURN",
    "COLUMN_TICKER",
    "DAYS_IN_MONTH",
    "DAYS_IN_YEAR",
    "PCT_TO_BPS",
    "SMALL_EPS",
    "TRADING_DAYS_IN_YEAR",
    "WEEKS_IN_YEAR",
    "annualisation_factor",
    "frequency_alias",
]


# Numeric constants ---------------------------------------------------------

DAYS_IN_YEAR: Final[int] = 365
"""Número de dias-calendário em um ano."""

TRADING_DAYS_IN_YEAR: Final[int] = 252
"""Número típico de pregões em mercados globais."""

WEEKS_IN_YEAR: Final[int] = 52
DAYS_IN_MONTH: Final[int] = 30

BPS: Final[float] = 1.0 / 10_000.0
"""Um *basis point* expresso em forma decimal."""

PCT_TO_BPS: Final[float] = 100.0
"""Fator para converter percentuais (0-100) em *basis points*."""

BPS_TO_PCT: Final[float] = 1.0 / PCT_TO_BPS
"""Fator para converter *basis points* em percentuais (0-100)."""

SMALL_EPS: Final[float] = 1e-9
"""Épsilon numérico padrão para comparações de ponto flutuante."""

DEFAULT_RISK_FREE_RATE: Final[float] = 0.05
"""Taxa livre de risco anualizada padrão (5%)."""

DEFAULT_BASE_CURRENCY: Final[str] = "BRL"
"""Moeda base adotada como padrão para relatórios e estratégias."""

# Column naming ------------------------------------------------------------

COLUMN_DATE: Final[str] = "date"
COLUMN_TICKER: Final[str] = "ticker"
COLUMN_PRICE: Final[str] = "price"
COLUMN_RETURN: Final[str] = "return"
COLUMN_ADV: Final[str] = "adv"


# Helper tables ------------------------------------------------------------

frequency_alias: Final[dict[str, str]] = {
    "d": "daily",
    "daily": "daily",
    "w": "weekly",
    "weekly": "weekly",
    "m": "monthly",
    "month": "monthly",
    "monthly": "monthly",
    "q": "quarterly",
    "quarterly": "quarterly",
    "y": "yearly",
    "year": "yearly",
    "yearly": "yearly",
}
"""Mapeia apelidos comuns para a frequência canonical."""

_annualisation_map: Final[dict[str, float]] = {
    "daily": TRADING_DAYS_IN_YEAR,
    "weekly": WEEKS_IN_YEAR,
    "monthly": 12.0,
    "quarterly": 4.0,
    "yearly": 1.0,
}


def annualisation_factor(frequency: str) -> float:
    """Return the factor used to annualise metrics from *frequency*.

    Parameters
    ----------
    frequency:
        Identificador (ex.: ``daily``, ``weekly``) ou qualquer apelido suportado
        em :data:`frequency_alias`.

    Raises
    ------
    KeyError
        Caso a frequência informada não seja suportada.
    """

    key = frequency_alias.get(frequency.lower(), frequency.lower())
    try:
        return float(_annualisation_map[key])
    except KeyError as exc:  # pragma: no cover - statement executed on failure
        raise KeyError(f"Unsupported frequency: {frequency}") from exc
