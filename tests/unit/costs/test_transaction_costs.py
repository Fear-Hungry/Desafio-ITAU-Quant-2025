import numpy as np
import pandas as pd
import pytest
from itau_quant.costs.transaction_costs import (
    bps_to_decimal,
    linear_transaction_cost,
    slippage_square_root_bps,
    slippage_transaction_cost,
    transaction_cost_vector,
)


def test_bps_to_decimal_handles_scalar_and_sequence():
    assert bps_to_decimal(25) == pytest.approx(0.0025)

    decimals = bps_to_decimal([10, 15])
    np.testing.assert_allclose(decimals, np.array([0.001, 0.0015]))


def test_linear_transaction_cost_returns_series_with_matching_index():
    idx = ["A", "B", "C"]
    weights = pd.Series([0.20, 0.30, 0.50], index=idx)
    prev_weights = pd.Series([0.10, 0.35, 0.55], index=idx)
    costs_bps = pd.Series([10, 20, 15], index=idx)

    per_asset = linear_transaction_cost(
        weights,
        prev_weights,
        costs_bps,
        notional=1_000_000,
        aggregate=False,
    )

    assert isinstance(per_asset, pd.Series)
    assert per_asset.index.tolist() == idx

    expected = np.abs(weights - prev_weights) * 1_000_000 * costs_bps / 10_000
    pd.testing.assert_series_equal(per_asset, expected.rename("linear_cost"))

    total = linear_transaction_cost(
        weights, prev_weights, costs_bps, notional=1_000_000
    )
    assert total == pytest.approx(expected.sum())


def test_slippage_transaction_cost_matches_square_root_model():
    weights = np.array([0.20, 0.30])
    prev_weights = np.array([0.15, 0.40])
    notional = 2_000_000
    adv = np.array([200_000, 100_000])
    coefficient = 40.0

    trades_notional = np.abs(weights - prev_weights) * notional
    impact = slippage_square_root_bps(
        trades_notional,
        adv,
        coefficient=coefficient,
        exponent=0.5,
    )

    expected_costs = trades_notional * np.asarray(impact) / 10_000

    realised = slippage_transaction_cost(
        weights,
        prev_weights,
        adv,
        notional=notional,
        coefficient=coefficient,
        exponent=0.5,
        aggregate=False,
    )

    np.testing.assert_allclose(realised, expected_costs)

    total = slippage_transaction_cost(
        weights,
        prev_weights,
        adv,
        notional=notional,
        coefficient=coefficient,
        exponent=0.5,
    )
    assert total == pytest.approx(expected_costs.sum())


def test_transaction_cost_vector_combines_linear_and_slippage():
    weights = np.array([0.25, 0.35])
    prev_weights = np.array([0.20, 0.40])
    linear_bps = np.array([5.0, 12.0])
    adv = np.array([250_000, 150_000])
    notional = 3_000_000

    linear_component = linear_transaction_cost(
        weights,
        prev_weights,
        linear_bps,
        notional=notional,
        aggregate=False,
    )
    slippage_component = slippage_transaction_cost(
        weights,
        prev_weights,
        adv,
        notional=notional,
        coefficient=30.0,
        exponent=0.5,
        aggregate=False,
    )

    combined = transaction_cost_vector(
        weights,
        prev_weights,
        linear_bps=linear_bps,
        adv=adv,
        notional=notional,
        coefficient=30.0,
        exponent=0.5,
    )

    np.testing.assert_allclose(combined, linear_component + slippage_component)


def test_slippage_transaction_cost_raises_when_adv_zero_with_trade():
    weights = np.array([0.30, 0.30])
    prev_weights = np.array([0.20, 0.30])
    adv = np.array([0.0, 100_000.0])

    with pytest.raises(ValueError, match="adv must be positive"):
        slippage_transaction_cost(weights, prev_weights, adv)
