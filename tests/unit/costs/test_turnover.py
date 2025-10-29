import numpy as np
import pandas as pd
import pytest

from itau_quant.costs import turnover as turnover_mod


def test_l1_turnover_returns_series_with_matching_index():
    idx = ["EWA", "EEM", "AGG"]
    weights = pd.Series([0.10, 0.20, 0.70], index=idx)
    prev = pd.Series([0.15, 0.22, 0.63], index=idx)

    per_asset = turnover_mod.l1_turnover(weights, prev, aggregate=False)

    assert isinstance(per_asset, pd.Series)
    assert per_asset.index.tolist() == idx
    pd.testing.assert_series_equal(
        per_asset,
        np.abs(weights - prev).rename("l1_turnover"),
    )

    total = turnover_mod.l1_turnover(weights, prev)
    assert total == pytest.approx(np.abs(weights - prev).sum())


def test_normalised_turnover_is_half_of_l1():
    weights = np.array([0.6, 0.4])
    prev = np.array([0.5, 0.5])

    l1_value = turnover_mod.l1_turnover(weights, prev)
    norm_value = turnover_mod.normalised_turnover(weights, prev)

    assert norm_value == pytest.approx(0.5 * l1_value)


def test_turnover_penalty_vector_and_aggregate():
    weights = np.array([0.4, 0.3, 0.3])
    prev = np.array([0.5, 0.2, 0.3])
    eta = 3.0

    vector = turnover_mod.turnover_penalty_vector(weights, prev, eta)
    np.testing.assert_allclose(vector, eta * np.abs(weights - prev))

    aggregate = turnover_mod.turnover_penalty_vector(
        weights,
        prev,
        eta,
        aggregate=True,
    )
    assert aggregate == pytest.approx(vector.sum())

    penalty = turnover_mod.turnover_penalty(weights, prev, eta)
    assert penalty == pytest.approx(vector.sum())


def test_turnover_violation_and_constraint_check():
    weights = np.array([0.25, 0.35, 0.40])
    prev = np.array([0.30, 0.30, 0.40])

    normalised = turnover_mod.normalised_turnover(weights, prev)
    tight_limit = normalised - 0.01
    loose_limit = normalised + 0.05

    violation = turnover_mod.turnover_violation(weights, prev, tight_limit)
    assert violation == pytest.approx(normalised - tight_limit)
    assert violation > 0

    assert not turnover_mod.is_within_turnover(weights, prev, tight_limit)
    assert turnover_mod.is_within_turnover(weights, prev, loose_limit)


def test_turnover_penalty_raises_on_negative_eta():
    weights = np.array([0.1, 0.9])
    prev = np.array([0.1, 0.9])

    with pytest.raises(ValueError, match="eta must be non-negative"):
        turnover_mod.turnover_penalty(weights, prev, -1.0)


def test_turnover_violation_raises_on_negative_limit():
    weights = np.array([0.1, 0.9])
    prev = np.array([0.2, 0.8])

    with pytest.raises(ValueError, match="max_turnover must be non-negative"):
        turnover_mod.turnover_violation(weights, prev, -0.1)
