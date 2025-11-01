from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from itau_quant.data.loader import (
    load_asset_prices,
    calculate_returns,
    download_and_cache_arara_prices,
    preprocess_data,
    download_and_preprocess_arara,
    download_fred_dtb3,
    DataLoader,
    DataBundle,
)


# =============================================================================
# load_asset_prices tests
# =============================================================================


def test_load_asset_prices_file_exists(tmp_path, monkeypatch):
    from itau_quant.data import loader as dl

    test_file = tmp_path / "test_prices.csv"
    index = pd.date_range("2020-01-01", periods=5, freq="D")
    prices = pd.DataFrame({"AAPL": [100, 110, 105, 115, 120]}, index=index)
    prices.to_csv(test_file, index=True)

    monkeypatch.setattr(dl, "RAW_DATA_DIR", tmp_path)

    result = load_asset_prices("test_prices.csv")
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 5
    assert "AAPL" in result.columns


def test_load_asset_prices_file_not_found(tmp_path, monkeypatch):
    from itau_quant.data import loader as dl

    monkeypatch.setattr(dl, "RAW_DATA_DIR", tmp_path)

    with pytest.raises(FileNotFoundError):
        load_asset_prices("nonexistent.csv")


def test_load_asset_prices_preserves_datetime_index(tmp_path, monkeypatch):
    from itau_quant.data import loader as dl

    test_file = tmp_path / "test_prices.csv"
    index = pd.date_range("2020-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"AAPL": [100, 110, 105]}, index=index)
    prices.to_csv(test_file, index=True)

    monkeypatch.setattr(dl, "RAW_DATA_DIR", tmp_path)

    result = load_asset_prices("test_prices.csv")
    assert isinstance(result.index, pd.DatetimeIndex)


# =============================================================================
# calculate_returns tests
# =============================================================================


def test_calculate_returns_delegates_to_processing_module():
    prices = pd.DataFrame({
        "AAPL": [100, 110, 105],
        "MSFT": [200, 210, 215]
    }, index=pd.date_range("2020-01-01", periods=3, freq="D"))

    result = calculate_returns(prices, method="log")
    assert isinstance(result, pd.DataFrame)
    assert result.shape[1] == prices.shape[1]
    assert len(result) > 0


def test_calculate_returns_default_method_is_log():
    prices = pd.DataFrame({
        "AAPL": [100, 110, 121]
    }, index=pd.date_range("2020-01-01", periods=3, freq="D"))

    result = calculate_returns(prices)
    assert result.iloc[0, 0] == pytest.approx(np.log(110/100), abs=1e-6)
    assert result.iloc[1, 0] == pytest.approx(np.log(121/110), abs=1e-6)


# =============================================================================
# download_and_cache_arara_prices tests
# =============================================================================


@patch("itau_quant.data.loader.yf_download")
@patch("itau_quant.data.loader.get_arara_universe")
def test_download_and_cache_arara_prices_creates_csv(
    mock_universe, mock_yf_download, tmp_path, monkeypatch
):
    from itau_quant.data import loader as dl

    monkeypatch.setattr(dl, "RAW_DATA_DIR", tmp_path)

    mock_universe.return_value = ["AAPL", "MSFT"]
    index = pd.date_range("2020-01-01", periods=3, freq="D")
    mock_prices = pd.DataFrame({
        "AAPL": [100, 110, 105],
        "MSFT": [200, 210, 215]
    }, index=index)
    mock_yf_download.return_value = mock_prices

    result_path = download_and_cache_arara_prices(
        start="2020-01-01",
        end="2020-01-03",
        raw_file_name="test_arara.csv"
    )

    assert result_path.exists()
    assert result_path.name == "test_arara.csv"

    loaded = pd.read_csv(result_path, index_col=0, parse_dates=True)
    assert loaded.shape == mock_prices.shape


@patch("itau_quant.data.loader.yf_download")
@patch("itau_quant.data.loader.get_arara_universe")
def test_download_and_cache_arara_prices_default_filename(
    mock_universe, mock_yf_download, tmp_path, monkeypatch
):
    from itau_quant.data import loader as dl

    monkeypatch.setattr(dl, "RAW_DATA_DIR", tmp_path)
    mock_universe.return_value = ["AAPL"]
    mock_yf_download.return_value = pd.DataFrame(
        {"AAPL": [100]}, index=pd.date_range("2020-01-01", periods=1)
    )

    result_path = download_and_cache_arara_prices()
    assert result_path.name == "prices_arara.csv"


# =============================================================================
# preprocess_data tests
# =============================================================================


def test_preprocess_data_creates_parquet(tmp_path, monkeypatch):
    from itau_quant.data import loader as dl

    raw_file = tmp_path / "raw_prices.csv"
    index = pd.date_range("2020-01-01", periods=5, freq="D")
    prices = pd.DataFrame({"AAPL": [100, 110, 105, 115, 120]}, index=index)
    prices.to_csv(raw_file, index=True)

    monkeypatch.setattr(dl, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(dl, "PROCESSED_DATA_DIR", tmp_path)

    result = preprocess_data("raw_prices.csv", "returns_test.parquet")

    assert isinstance(result, pd.DataFrame)
    assert (tmp_path / "returns_test.parquet").exists()


def test_preprocess_data_returns_dataframe_with_returns(tmp_path, monkeypatch):
    from itau_quant.data import loader as dl

    raw_file = tmp_path / "raw_prices.csv"
    index = pd.date_range("2020-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"AAPL": [100, 110, 121]}, index=index)
    prices.to_csv(raw_file, index=True)

    monkeypatch.setattr(dl, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(dl, "PROCESSED_DATA_DIR", tmp_path)

    result = preprocess_data("raw_prices.csv", "returns_test.parquet")

    assert result.iloc[0, 0] == pytest.approx(np.log(110/100), abs=1e-6)
    assert result.iloc[1, 0] == pytest.approx(np.log(121/110), abs=1e-6)


# =============================================================================
# download_and_preprocess_arara tests
# =============================================================================


@patch("itau_quant.data.loader.download_and_cache_arara_prices")
@patch("itau_quant.data.loader.preprocess_data")
def test_download_and_preprocess_arara_chains_functions(
    mock_preprocess, mock_download, tmp_path
):
    mock_path = tmp_path / "prices_arara.csv"
    mock_download.return_value = mock_path

    expected_returns = pd.DataFrame({"AAPL": [0.01, 0.02]})
    mock_preprocess.return_value = expected_returns

    result = download_and_preprocess_arara(
        start="2020-01-01",
        end="2020-01-03",
        processed_file_name="returns_test.parquet"
    )

    mock_download.assert_called_once_with(start="2020-01-01", end="2020-01-03")
    mock_preprocess.assert_called_once_with("prices_arara.csv", "returns_test.parquet")
    pd.testing.assert_frame_equal(result, expected_returns)


# =============================================================================
# download_fred_dtb3 tests
# =============================================================================


@patch("itau_quant.data.loader.fred_download_dtb3")
def test_download_fred_dtb3_delegates_to_sources_module(mock_fred):
    expected_series = pd.Series([0.01, 0.02, 0.015], index=pd.date_range("2020-01-01", periods=3))
    mock_fred.return_value = expected_series

    result = download_fred_dtb3(start="2020-01-01", end="2020-01-03")

    mock_fred.assert_called_once_with(start="2020-01-01", end="2020-01-03")
    pd.testing.assert_series_equal(result, expected_series)


# =============================================================================
# DataBundle tests
# =============================================================================


def test_databundle_is_frozen():
    index = pd.date_range("2020-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"AAPL": [100, 110, 120]}, index=index)
    returns = pd.DataFrame({"AAPL": [0.0, 0.1, 0.09]}, index=index)
    rf = pd.Series([0.001, 0.001, 0.001], index=index)
    excess = pd.DataFrame({"AAPL": [0.0, 0.099, 0.089]}, index=index)
    bms = index
    inception = pd.Series({"AAPL": index[0]})

    bundle = DataBundle(
        prices=prices,
        returns=returns,
        rf_daily=rf,
        excess_returns=excess,
        bms=bms,
        inception_mask=inception
    )

    with pytest.raises(AttributeError):
        bundle.prices = pd.DataFrame()


def test_databundle_fields():
    index = pd.date_range("2020-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"AAPL": [100, 110, 120]}, index=index)
    returns = pd.DataFrame({"AAPL": [0.0, 0.1, 0.09]}, index=index)
    rf = pd.Series([0.001, 0.001, 0.001], index=index)
    excess = pd.DataFrame({"AAPL": [0.0, 0.099, 0.089]}, index=index)
    bms = index
    inception = pd.Series({"AAPL": index[0]})

    bundle = DataBundle(
        prices=prices,
        returns=returns,
        rf_daily=rf,
        excess_returns=excess,
        bms=bms,
        inception_mask=inception
    )

    assert bundle.prices.equals(prices)
    assert bundle.returns.equals(returns)
    assert bundle.rf_daily.equals(rf)
    assert bundle.excess_returns.equals(excess)
    assert bundle.bms.equals(bms)
    assert bundle.inception_mask.equals(inception)


# =============================================================================
# DataLoader.__init__ tests
# =============================================================================


def test_dataloader_init_default_tickers():
    with patch("itau_quant.data.loader.get_arara_universe") as mock_universe:
        mock_universe.return_value = ["AAPL", "MSFT", "GOOG"]
        loader = DataLoader()
        assert loader.tickers == ["AAPL", "MSFT", "GOOG"]


def test_dataloader_init_custom_tickers():
    loader = DataLoader(tickers=["AAPL", "MSFT"])
    assert loader.tickers == ["AAPL", "MSFT"]


def test_dataloader_init_parameters():
    loader = DataLoader(
        tickers=["AAPL"],
        start="2020-01-01",
        end="2020-12-31",
        mode="BMS"
    )
    assert loader.start == "2020-01-01"
    assert loader.end == "2020-12-31"
    assert loader.mode == "BMS"
    assert loader.actions is None


def test_dataloader_init_with_actions():
    actions = [
        {"ticker": "AAPL", "event_type": "split", "ex_date": "2020-08-31", "ratio": 4.0}
    ]
    loader = DataLoader(tickers=["AAPL"], actions=actions)
    assert loader.actions == actions


# =============================================================================
# DataLoader.load tests
# =============================================================================


@patch("itau_quant.data.loader.yf_download")
@patch("itau_quant.data.loader.normalize_index")
@patch("itau_quant.data.loader.filter_liquid_assets")
@patch("itau_quant.data.loader.validate_panel")
@patch("itau_quant.data.loader._calculate_returns")
@patch("itau_quant.data.loader.fred_download_dtb3")
@patch("itau_quant.data.loader.compute_excess_returns")
@patch("itau_quant.data.loader.rebalance_schedule")
@patch("itau_quant.data.loader.request_hash")
@patch("itau_quant.data.loader.save_parquet")
def test_dataloader_load_without_corporate_actions(
    mock_save,
    mock_hash,
    mock_rebalance,
    mock_excess,
    mock_fred,
    mock_returns,
    mock_validate,
    mock_filter,
    mock_normalize,
    mock_yf
):
    index = pd.date_range("2020-01-01", periods=5, freq="D")
    prices = pd.DataFrame({"AAPL": [100, 110, 105, 115, 120]}, index=index)
    returns = pd.DataFrame({"AAPL": [0.0, 0.095, -0.047, 0.087, 0.042]}, index=index)
    rf = pd.Series([0.001] * 5, index=index)
    excess = returns - rf.values.reshape(-1, 1)
    bms = pd.DatetimeIndex([index[0], index[2], index[4]])

    mock_yf.return_value = prices
    mock_normalize.return_value = prices
    liquidity = pd.DataFrame(
        {
            "is_liquid": [True],
            "coverage": [1.0],
            "non_na": [len(index)],
            "max_gap": [0],
            "first_valid": [index[0]],
            "last_valid": [index[-1]],
        },
        index=["AAPL"],
    )
    mock_filter.return_value = (prices, liquidity)
    mock_returns.return_value = returns
    mock_fred.return_value = rf
    mock_excess.return_value = excess
    mock_rebalance.return_value = bms
    mock_hash.return_value = "test_hash"

    loader = DataLoader(tickers=["AAPL"], start="2020-01-01", end="2020-01-05")
    bundle = loader.load()

    assert isinstance(bundle, DataBundle)
    assert bundle.prices.equals(prices)
    assert bundle.returns.equals(returns)
    assert bundle.rf_daily.equals(rf)
    assert bundle.bms.equals(bms)
    assert len(bundle.inception_mask) == 1
    artefacts = loader.artifacts
    assert artefacts["request_id"] == "test_hash"
    assert artefacts["returns_path"].name.endswith("returns_test_hash.parquet")
    assert artefacts["prices_path"].name.endswith("prices_test_hash.parquet")
    assert artefacts["rf_path"].name.endswith("rf_daily_test_hash.parquet")


@patch("itau_quant.data.loader.yf_download")
@patch("itau_quant.data.loader.normalize_index")
@patch("itau_quant.data.loader.load_corporate_actions")
@patch("itau_quant.data.loader.calculate_adjustment_factors")
@patch("itau_quant.data.loader.apply_price_adjustments")
@patch("itau_quant.data.loader.filter_liquid_assets")
@patch("itau_quant.data.loader.validate_panel")
@patch("itau_quant.data.loader._calculate_returns")
@patch("itau_quant.data.loader.fred_download_dtb3")
@patch("itau_quant.data.loader.compute_excess_returns")
@patch("itau_quant.data.loader.rebalance_schedule")
@patch("itau_quant.data.loader.request_hash")
@patch("itau_quant.data.loader.save_parquet")
def test_dataloader_load_with_corporate_actions(
    mock_save,
    mock_hash,
    mock_rebalance,
    mock_excess,
    mock_fred,
    mock_returns,
    mock_validate,
    mock_filter,
    mock_adjust,
    mock_factors,
    mock_load_actions,
    mock_normalize,
    mock_yf
):
    index = pd.date_range("2020-01-01", periods=5, freq="D")
    raw_prices = pd.DataFrame({"AAPL": [100, 50, 50, 55, 60]}, index=index)
    adjusted_prices = pd.DataFrame({"AAPL": [100, 100, 100, 110, 120]}, index=index)
    returns = pd.DataFrame({"AAPL": [0.0, 0.0, 0.0, 0.095, 0.087]}, index=index)
    rf = pd.Series([0.001] * 5, index=index)
    excess = returns - rf.values.reshape(-1, 1)
    bms = pd.DatetimeIndex([index[0], index[2], index[4]])

    actions_list = [
        {"ticker": "AAPL", "event_type": "split", "ex_date": "2020-01-02", "ratio": 2.0}
    ]
    actions_df = pd.DataFrame(actions_list)
    factors_df = pd.DataFrame({"price_AAPL": [1.0, 0.5, 0.5, 0.5, 0.5]}, index=index)

    mock_yf.return_value = raw_prices
    mock_normalize.return_value = raw_prices
    mock_load_actions.return_value = actions_df
    mock_factors.return_value = factors_df
    mock_adjust.return_value = adjusted_prices
    mock_filter.return_value = (
        adjusted_prices,
        pd.DataFrame(
            {
                "is_liquid": [True],
                "coverage": [1.0],
                "non_na": [len(index)],
                "max_gap": [0],
                "first_valid": [index[0]],
                "last_valid": [index[-1]],
            },
            index=["AAPL"],
        ),
    )
    mock_returns.return_value = returns
    mock_fred.return_value = rf
    mock_excess.return_value = excess
    mock_rebalance.return_value = bms
    mock_hash.return_value = "test_hash"

    loader = DataLoader(tickers=["AAPL"], start="2020-01-01", end="2020-01-05", actions=actions_list)
    bundle = loader.load()

    assert isinstance(bundle, DataBundle)
    mock_load_actions.assert_called_once()
    mock_factors.assert_called_once()
    mock_adjust.assert_called_once()


@patch("itau_quant.data.loader.yf_download")
@patch("itau_quant.data.loader.normalize_index")
@patch("itau_quant.data.loader.filter_liquid_assets")
def test_dataloader_load_filters_illiquid_assets(mock_filter, mock_normalize, mock_yf):
    index = pd.date_range("2020-01-01", periods=5, freq="D")
    prices = pd.DataFrame({
        "AAPL": [100, 110, 105, 115, 120],
        "ILLIQUID": [None, None, 50, None, None]
    }, index=index)

    filtered_prices = pd.DataFrame({"AAPL": [100, 110, 105, 115, 120]}, index=index)
    stats = pd.DataFrame(
        {
            "is_liquid": [True, False],
            "coverage": [1.0, 0.2],
            "non_na": [len(index), 1],
            "max_gap": [0, len(index)],
            "first_valid": [index[0], index[2]],
            "last_valid": [index[-1], index[2]],
        },
        index=["AAPL", "ILLIQUID"],
    )

    mock_yf.return_value = prices
    mock_normalize.return_value = prices
    mock_filter.return_value = (filtered_prices, stats)

    with patch("itau_quant.data.loader.validate_panel"):
        with patch("itau_quant.data.loader._calculate_returns"):
            with patch("itau_quant.data.loader.fred_download_dtb3"):
                with patch("itau_quant.data.loader.compute_excess_returns"):
                    with patch("itau_quant.data.loader.rebalance_schedule"):
                        with patch("itau_quant.data.loader.request_hash"):
                            with patch("itau_quant.data.loader.save_parquet"):
                                loader = DataLoader(tickers=["AAPL", "ILLIQUID"])
                                bundle = loader.load()

                                assert "AAPL" in bundle.prices.columns
                                assert "ILLIQUID" not in bundle.prices.columns


@patch("itau_quant.data.loader.yf_download")
@patch("itau_quant.data.loader.normalize_index")
@patch("itau_quant.data.loader.filter_liquid_assets")
def test_dataloader_load_raises_when_no_liquid_assets(mock_filter, mock_normalize, mock_yf):
    index = pd.date_range("2020-01-01", periods=5, freq="D")
    prices = pd.DataFrame({"ILLIQUID": [None, None, 50, None, None]}, index=index)

    empty_prices = pd.DataFrame(index=index)
    stats = pd.DataFrame(
        {
            "is_liquid": [False],
            "coverage": [0.2],
            "non_na": [1],
            "max_gap": [len(index)],
            "first_valid": [index[2]],
            "last_valid": [index[2]],
        },
        index=["ILLIQUID"],
    )

    mock_yf.return_value = prices
    mock_normalize.return_value = prices
    mock_filter.return_value = (empty_prices, stats)

    loader = DataLoader(tickers=["ILLIQUID"])

    with pytest.raises(ValueError, match="Nenhum ativo restante após filtros de liquidez"):
        loader.load()


@patch("itau_quant.data.loader.yf_download")
@patch("itau_quant.data.loader.normalize_index")
@patch("itau_quant.data.loader.filter_liquid_assets")
@patch("itau_quant.data.loader.validate_panel")
@patch("itau_quant.data.loader._calculate_returns")
@patch("itau_quant.data.loader.fred_download_dtb3")
@patch("itau_quant.data.loader.compute_excess_returns")
@patch("itau_quant.data.loader.rebalance_schedule")
@patch("itau_quant.data.loader.request_hash")
@patch("itau_quant.data.loader.save_parquet")
def test_dataloader_load_saves_parquet_files(
    mock_save,
    mock_hash,
    mock_rebalance,
    mock_excess,
    mock_fred,
    mock_returns,
    mock_validate,
    mock_filter,
    mock_normalize,
    mock_yf
):
    index = pd.date_range("2020-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"AAPL": [100, 110, 120]}, index=index)
    returns = pd.DataFrame({"AAPL": [0.0, 0.095, 0.087]}, index=index)
    rf = pd.Series([0.001] * 3, index=index)
    excess = returns - rf.values.reshape(-1, 1)

    mock_yf.return_value = prices
    mock_normalize.return_value = prices
    mock_filter.return_value = (
        prices,
        pd.DataFrame(
            {
                "is_liquid": [True],
                "coverage": [1.0],
                "non_na": [len(index)],
                "max_gap": [0],
                "first_valid": [index[0]],
                "last_valid": [index[-1]],
            },
            index=["AAPL"],
        ),
    )
    mock_returns.return_value = returns
    mock_fred.return_value = rf
    mock_excess.return_value = excess
    mock_rebalance.return_value = index
    mock_hash.return_value = "abc123"

    loader = DataLoader(tickers=["AAPL"])
    loader.load()

    assert mock_save.call_count == 4
    save_calls = [str(call[0][0]) for call in mock_save.call_args_list]
    assert any("prices_abc123.parquet" in path for path in save_calls)
    assert any("returns_abc123.parquet" in path for path in save_calls)
    assert any("excess_returns_abc123.parquet" in path for path in save_calls)
    assert any("rf_daily_abc123.parquet" in path for path in save_calls)


@patch("itau_quant.data.loader.crypto_download")
@patch("itau_quant.data.loader.yf_download")
@patch("itau_quant.data.loader.normalize_index")
@patch("itau_quant.data.loader.filter_liquid_assets")
@patch("itau_quant.data.loader.validate_panel")
@patch("itau_quant.data.loader._calculate_returns")
@patch("itau_quant.data.loader.fred_download_dtb3")
@patch("itau_quant.data.loader.compute_excess_returns")
@patch("itau_quant.data.loader.rebalance_schedule")
@patch("itau_quant.data.loader.request_hash")
@patch("itau_quant.data.loader.save_parquet")
def test_dataloader_uses_crypto_connector(
    mock_save,
    mock_hash,
    mock_rebalance,
    mock_excess,
    mock_fred,
    mock_returns,
    mock_validate,
    mock_filter,
    mock_normalize,
    mock_yf,
    mock_crypto,
    monkeypatch,
):
    index = pd.date_range("2024-01-01", periods=3, freq="D")

    multi_cols = pd.MultiIndex.from_product([["close"], ["IBIT"]], names=["field", "symbol"])
    crypto_panel = pd.DataFrame([[100.0], [101.0], [103.0]], index=index, columns=multi_cols)
    normalized_crypto = pd.DataFrame({"IBIT": [100.0, 101.0, 103.0]}, index=index)
    returns = pd.DataFrame({"IBIT": [0.0, 0.0099, 0.0197]}, index=index)
    rf = pd.Series([0.0001] * len(index), index=index)
    excess = returns - rf.values.reshape(-1, 1)

    from itau_quant.data import loader as dl

    monkeypatch.setattr(dl, "get_arara_metadata", lambda: {"IBIT": {"asset_class": "Crypto"}})

    mock_crypto.return_value = crypto_panel
    mock_yf.side_effect = AssertionError("yf_download should not be called for crypto tickers")
    mock_normalize.side_effect = lambda df: df
    mock_filter.return_value = (
        normalized_crypto,
        pd.DataFrame(
            {
                "is_liquid": [True],
                "coverage": [1.0],
                "non_na": [len(index)],
                "max_gap": [0],
                "first_valid": [index[0]],
                "last_valid": [index[-1]],
            },
            index=["IBIT"],
        ),
    )
    mock_returns.return_value = returns
    mock_fred.return_value = rf
    mock_excess.return_value = excess
    mock_rebalance.return_value = index
    mock_hash.return_value = "crypto123"

    loader = DataLoader(tickers=["IBIT"], start="2024-01-01", end="2024-01-10")
    bundle = loader.load()

    pd.testing.assert_frame_equal(bundle.prices, normalized_crypto)
    mock_crypto.assert_called_once_with(
        ["IBIT"], start="2024-01-01", end="2024-01-10", cache=True, force_refresh=False
    )
    assert mock_save.call_count == 4
    artefacts = loader.artifacts
    assert artefacts["metadata"]["liquidity"]["liquid"] == 1

def test_dataloader_load_reuses_cache(tmp_path, monkeypatch):
    from itau_quant.data import loader as dl

    index = pd.date_range("2020-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"AAA": [100.0, 101.0, 102.0]}, index=index)
    returns = pd.DataFrame({"AAA": [0.0, 0.01, 0.009]}, index=index)
    rf = pd.Series([0.001] * 3, index=index, name="rf_daily")
    excess = returns - 0.001

    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    monkeypatch.setattr(dl, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(dl, "PROCESSED_DATA_DIR", processed_dir)
    monkeypatch.setattr(dl, "normalize_index", lambda df: df)
    def fake_filter(df, **_):
        index_local = df.index
        stats = pd.DataFrame(
            {
                "is_liquid": [True] * len(df.columns),
                "coverage": [1.0] * len(df.columns),
                "non_na": [len(index_local)] * len(df.columns),
                "max_gap": [0] * len(df.columns),
                "first_valid": [index_local[0]] * len(df.columns),
                "last_valid": [index_local[-1]] * len(df.columns),
            },
            index=df.columns,
        )
        return df, stats

    monkeypatch.setattr(dl, "filter_liquid_assets", fake_filter)
    monkeypatch.setattr(dl, "validate_panel", lambda df: None)
    monkeypatch.setattr(dl, "_calculate_returns", lambda df, method="log": returns.copy())
    monkeypatch.setattr(dl, "fred_download_dtb3", lambda start, end: rf.copy())
    monkeypatch.setattr(dl, "compute_excess_returns", lambda ret, rf_series: excess.copy())
    monkeypatch.setattr(dl, "rebalance_schedule", lambda idx, mode: idx)
    monkeypatch.setattr(dl, "request_hash", lambda tickers, start, end: "cachehash")

    download_calls = {"count": 0}

    def fake_yf(tickers, start, end):
        download_calls["count"] += 1
        return prices.copy()

    monkeypatch.setattr(dl, "yf_download", fake_yf)

    loader = DataLoader(tickers=["AAA"], start="2020-01-01", end="2020-01-03")
    loader.load()
    assert download_calls["count"] == 1
    assert (processed_dir / "returns_cachehash.parquet").exists()

    def fail_yf(*_args, **_kwargs):
        raise AssertionError("cache miss triggered unexpected download")

    monkeypatch.setattr(dl, "yf_download", fail_yf)

    loader_cached = DataLoader(tickers=["AAA"], start="2020-01-01", end="2020-01-03")
    cached_bundle = loader_cached.load()

    assert cached_bundle.prices.equals(prices)
    assert loader_cached.artifacts["from_cache"] is True
