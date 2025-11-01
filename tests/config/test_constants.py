import pytest
from itau_quant.config.constants import (
    BPS,
    DEFAULT_BASE_CURRENCY,
    DEFAULT_RISK_FREE_RATE,
    SMALL_EPS,
    TRADING_DAYS_IN_YEAR,
    annualisation_factor,
)


def test_annualisation_factor_supports_aliases():
    assert annualisation_factor("daily") == pytest.approx(TRADING_DAYS_IN_YEAR)
    assert annualisation_factor("d") == pytest.approx(TRADING_DAYS_IN_YEAR)
    assert annualisation_factor("Monthly") == pytest.approx(12.0)
    with pytest.raises(KeyError):
        annualisation_factor("intraday")


def test_core_constants_are_reasonable():
    assert BPS == pytest.approx(0.0001)
    assert DEFAULT_RISK_FREE_RATE > 0
    assert SMALL_EPS < 1e-6
    assert DEFAULT_BASE_CURRENCY == "BRL"
