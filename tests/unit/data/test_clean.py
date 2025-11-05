from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from arara_quant.data.processing.clean import (
    ensure_dtindex,
    filter_liquid_assets,
    normalize_index,
    validate_panel,
    winsorize_outliers,
)


def test_ensure_dtindex_normalizes_and_sorts():
    idx = ["2024-01-02", "2024-01-01", "2024-01-01"]
    out = ensure_dtindex(idx)
    assert isinstance(out, pd.DatetimeIndex)
    assert out.is_monotonic_increasing
    assert out.is_unique


def test_normalize_index_dataframe():
    df = pd.DataFrame({"A": [1, 2]}, index=["2024-01-02", "2024-01-01"])
    out = normalize_index(df)
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.is_monotonic_increasing


def test_normalize_index_preserves_value_alignment():
    df = pd.DataFrame(
        {"A": [3, 1, 2], "B": [30, 10, 20]},
        index=["2024-01-03", "2024-01-01", "2024-01-02"],
    )

    out = normalize_index(df)

    expected_index = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"])
    assert out.index.equals(expected_index)
    pd.testing.assert_series_equal(
        out.loc["2024-01-01"],
        pd.Series({"A": 1, "B": 10}, name=pd.Timestamp("2024-01-01")),
    )
    pd.testing.assert_series_equal(
        out.loc["2024-01-02"],
        pd.Series({"A": 2, "B": 20}, name=pd.Timestamp("2024-01-02")),
    )
    pd.testing.assert_series_equal(
        out.loc["2024-01-03"],
        pd.Series({"A": 3, "B": 30}, name=pd.Timestamp("2024-01-03")),
    )


def test_normalize_index_rejects_duplicate_dates():
    df = pd.DataFrame(
        {"A": [1.0, 2.0]},
        index=["2024-01-01", "2024-01-01"],
    )
    with pytest.raises(ValueError):
        normalize_index(df)


def test_normalize_index_handles_timezone_and_ordering():
    tz_index = pd.DatetimeIndex(
        ["2024-01-02 09:30", "2024-01-01 09:30"], tz="America/New_York"
    )
    df = pd.DataFrame({"A": [2, 1]}, index=tz_index)

    out = normalize_index(df)

    assert out.index.tz is None
    expected_index = pd.DatetimeIndex(["2024-01-01 09:30", "2024-01-02 09:30"])
    assert out.index.equals(expected_index)
    assert out.loc[pd.Timestamp("2024-01-01 09:30"), "A"] == 1
    assert out.loc[pd.Timestamp("2024-01-02 09:30"), "A"] == 2


def test_validate_panel_basic_checks():
    df = pd.DataFrame({"A": [1.0, 2.0]}, index=pd.bdate_range("2024-01-01", periods=2))
    validate_panel(df)  # não deve lançar


def test_filter_liquid_assets_removes_sparse_assets():
    idx = pd.bdate_range("2024-01-01", periods=10)
    df = pd.DataFrame(
        {
            "LIQ": [float(i) for i in range(10)],
            "GAPS": [1.0, None, None, None, 5.0, 6.0, 7.0, None, None, None],
            "SHORT": [None, None, 2.0, 3.0, None, 5.0, None, None, None, None],
        },
        index=idx,
    )

    filtered, stats = filter_liquid_assets(
        df, min_history=6, min_coverage=0.7, max_gap=2
    )

    assert list(filtered.columns) == ["LIQ"]
    assert stats.loc["LIQ", "is_liquid"]
    assert not stats.loc["GAPS", "is_liquid"]
    assert not stats.loc["SHORT", "is_liquid"]


def test_filter_liquid_assets_respects_per_asset_override():
    idx = pd.bdate_range("2023-01-02", periods=300)
    df = pd.DataFrame({"FULL": range(300)}, index=idx, dtype=float)
    short = pd.Series(float("nan"), index=idx)
    short.iloc[-80:] = range(80)
    df["SHORT"] = short

    filtered_default, _ = filter_liquid_assets(df)
    assert list(filtered_default.columns) == ["FULL"]

    filtered_override, stats_override = filter_liquid_assets(
        df,
        per_asset_min_history={"SHORT": 60},
    )

    assert "SHORT" in filtered_override.columns
    assert stats_override.loc["SHORT", "is_liquid"]


def test_winsorize_outliers_series():
    series = pd.Series([1.0, 2.0, 100.0, 3.0, -50.0])
    out = winsorize_outliers(series, lower=0.1, upper=0.9)
    assert out.max() == pytest.approx(series.quantile(0.9))
    assert out.min() == pytest.approx(series.quantile(0.1))


def test_winsorize_outliers_dataframe_global_bounds():
    df = pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 100.0],
            "B": [-10.0, -8.0, -6.0, 50.0],
        }
    )
    out = winsorize_outliers(df, lower=0.25, upper=0.75, per_column=False)
    stacked = df.stack()
    lower_bound = stacked.quantile(0.25)
    upper_bound = stacked.quantile(0.75)
    assert (out["A"] <= upper_bound + 1e-9).all()
    assert (out["B"] >= lower_bound - 1e-9).all()


def test_winsorize_outliers_invalid_params():
    with pytest.raises(ValueError):
        winsorize_outliers(pd.Series([1, 2, 3]), lower=0.9, upper=0.1)
