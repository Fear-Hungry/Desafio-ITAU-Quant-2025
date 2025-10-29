from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from itau_quant.backtesting.risk_monitor import apply_turnover_cap, evaluate_turnover_band


def test_apply_turnover_cap_returns_target_when_within_limit() -> None:
    prev = pd.Series([0.25, 0.75], index=["AAA", "BBB"])
    target = pd.Series([0.30, 0.70], index=["AAA", "BBB"])

    adjusted, turnover = apply_turnover_cap(prev, target, max_turnover=0.20)

    pd.testing.assert_series_equal(adjusted, target.astype(float))
    assert turnover == pytest.approx(np.abs(target - prev).sum())


def test_apply_turnover_cap_scales_when_exceeds_limit() -> None:
    prev = pd.Series([0.40, 0.60], index=["AAA", "BBB"])
    target = pd.Series([0.10, 0.90], index=["AAA", "BBB"])

    adjusted, turnover = apply_turnover_cap(prev, target, max_turnover=0.30)

    scale = 0.30 / np.abs(target - prev).sum()
    expected = prev + (target - prev) * scale
    pd.testing.assert_series_equal(adjusted, expected.astype(float))
    assert turnover == pytest.approx(0.30)


def test_apply_turnover_cap_aligns_missing_assets() -> None:
    prev = pd.Series({"AAA": 0.30, "BBB": 0.40, "CCC": 0.30})
    target = pd.Series({"BBB": 0.35, "DDD": 0.65})

    adjusted, turnover = apply_turnover_cap(prev, target, max_turnover=None)

    pd.testing.assert_index_equal(adjusted.index, pd.Index(["BBB", "DDD"]))
    aligned_prev = prev.reindex(adjusted.index, fill_value=0.0)
    aligned_target = target.reindex(adjusted.index, fill_value=0.0)
    expected_turnover = float(np.abs(aligned_target - aligned_prev).sum())
    assert turnover == pytest.approx(expected_turnover)


def test_apply_turnover_cap_raises_on_invalid_inputs() -> None:
    prev = pd.Series([0.5, 0.5], index=["AAA", "BBB"])
    target = pd.Series([0.6, np.nan], index=["AAA", "BBB"])

    with pytest.raises(ValueError, match="weights must not contain NaN values"):
        apply_turnover_cap(prev, target, max_turnover=0.10)

    target_clean = target.fillna(0.4)
    with pytest.raises(ValueError, match="max_turnover must be non-negative"):
        apply_turnover_cap(prev, target_clean, max_turnover=-0.1)


def test_evaluate_turnover_band_classifies_relative_to_band() -> None:
    band = (0.10, 0.20)
    assert evaluate_turnover_band(0.05, band) == "below"
    assert evaluate_turnover_band(0.15, band) == "within"
    assert evaluate_turnover_band(0.30, band) == "above"
    assert evaluate_turnover_band(0.12, None) == "within"


def test_evaluate_turnover_band_requires_two_values() -> None:
    with pytest.raises(ValueError, match="turnover band must have exactly two values"):
        evaluate_turnover_band(0.10, [0.05])
