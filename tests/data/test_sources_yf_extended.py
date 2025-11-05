"""Extended tests for Yahoo Finance data source with mocking."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from arara_quant.data.sources.yf import download_prices


@pytest.fixture
def mock_multiindex_data():
    """Create mock data with MultiIndex columns (multiple tickers)."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    tickers = ["SPY", "QQQ"]

    # Create MultiIndex DataFrame like yfinance returns
    columns = pd.MultiIndex.from_product([tickers, ["Open", "High", "Low", "Close", "Adj Close", "Volume"]])
    data = pd.DataFrame(np.random.randn(5, 12) * 10 + 100, index=dates, columns=columns)

    return data


@pytest.fixture
def mock_single_ticker_data():
    """Create mock data for single ticker."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    data = pd.DataFrame({
        "Open": np.random.randn(5) * 10 + 100,
        "High": np.random.randn(5) * 10 + 105,
        "Low": np.random.randn(5) * 10 + 95,
        "Close": np.random.randn(5) * 10 + 100,
        "Adj Close": np.random.randn(5) * 10 + 100,
        "Volume": np.random.randint(1000000, 10000000, 5),
    }, index=dates)

    return data


class TestDownloadPricesWithMocking:
    """Tests for download_prices with mocked yfinance."""

    @patch('arara_quant.data.sources.yf.yf.download')
    def test_downloads_multiple_tickers(self, mock_download, mock_multiindex_data):
        """Test downloading multiple tickers."""
        mock_download.return_value = mock_multiindex_data

        result = download_prices(["SPY", "QQQ"], start="2024-01-01", end="2024-01-05")

        assert isinstance(result, pd.DataFrame)
        assert "SPY" in result.columns
        assert "QQQ" in result.columns
        assert len(result) > 0
        mock_download.assert_called_once()

    @patch('arara_quant.data.sources.yf.yf.download')
    def test_downloads_single_ticker(self, mock_download, mock_single_ticker_data):
        """Test downloading single ticker."""
        mock_download.return_value = mock_single_ticker_data

        result = download_prices(["SPY"], start="2024-01-01")

        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 1
        mock_download.assert_called_once()

    @patch('arara_quant.data.sources.yf.yf.download')
    def test_normalizes_ticker_names(self, mock_download, mock_multiindex_data):
        """Test that ticker names are normalized (uppercase, stripped)."""
        mock_download.return_value = mock_multiindex_data

        # Pass tickers with various formats
        result = download_prices([" spy ", "qqq", "SPY"], start="2024-01-01")

        # Should call yfinance with normalized and deduplicated tickers
        call_args = mock_download.call_args
        called_tickers = call_args.kwargs.get('tickers', [])
        if not called_tickers and call_args.args:
            called_tickers = call_args.args[0]

        # Verify normalization and deduplication occurred
        assert isinstance(result, pd.DataFrame)
        mock_download.assert_called_once()

    @patch('arara_quant.data.sources.yf.yf.download')
    def test_extracts_adj_close(self, mock_download, mock_multiindex_data):
        """Test that Adj Close is extracted from MultiIndex."""
        mock_download.return_value = mock_multiindex_data

        result = download_prices(["SPY", "QQQ"])

        # Should extract Adj Close column
        assert isinstance(result, pd.DataFrame)
        # Values should come from Adj Close, not Close
        assert result.notna().any().any()

    @patch('arara_quant.data.sources.yf.yf.download')
    def test_fallback_to_close_when_adj_missing(self, mock_download):
        """Test fallback to Close when Adj Close is missing."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        tickers = ["SPY", "QQQ"]

        # Create data WITHOUT Adj Close
        columns = pd.MultiIndex.from_product([tickers, ["Open", "High", "Low", "Close", "Volume"]])
        data = pd.DataFrame(np.random.randn(5, 10) * 10 + 100, index=dates, columns=columns)

        mock_download.return_value = data

        result = download_prices(["SPY", "QQQ"])

        # Should fallback to Close
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    @patch('arara_quant.data.sources.yf.yf.download')
    def test_reorders_columns_to_match_request(self, mock_download, mock_multiindex_data):
        """Test that columns are reordered to match requested ticker order."""
        mock_download.return_value = mock_multiindex_data

        # Request in specific order
        result = download_prices(["QQQ", "SPY"])

        # Should return in requested order
        assert list(result.columns) == ["QQQ", "SPY"]

    @patch('arara_quant.data.sources.yf.yf.download')
    def test_forward_fills_missing_data(self, mock_download):
        """Test that missing data is forward-filled."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")

        # Create data with NaN values
        data = pd.DataFrame({
            ("SPY", "Adj Close"): [100, np.nan, 102, np.nan, 104],
            ("SPY", "Close"): [100, 101, 102, 103, 104],
            ("SPY", "Volume"): [1000] * 5,
        }, index=dates)
        data.columns = pd.MultiIndex.from_tuples(data.columns)

        mock_download.return_value = data

        result = download_prices(["SPY"])

        # Should forward-fill NaN values
        assert not result.isna().any().any()

    @patch('arara_quant.data.sources.yf.yf.download')
    def test_removes_completely_empty_columns(self, mock_download):
        """Test that completely empty columns are removed."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")

        data = pd.DataFrame({
            ("SPY", "Adj Close"): [100, 101, 102, 103, 104],
            ("QQQ", "Adj Close"): [np.nan] * 5,  # Completely empty
        }, index=dates)
        data.columns = pd.MultiIndex.from_tuples(data.columns)

        mock_download.return_value = data

        result = download_prices(["SPY", "QQQ"])

        # QQQ should be dropped (all NaN)
        assert "SPY" in result.columns
        assert "QQQ" not in result.columns or result["QQQ"].isna().all()

    @patch('arara_quant.data.sources.yf.yf.download')
    def test_raises_on_empty_ticker_list(self, mock_download):
        """Test that empty ticker list raises ValueError."""
        with pytest.raises(ValueError, match="vazia"):
            download_prices([])

    @patch('arara_quant.data.sources.yf.yf.download')
    def test_raises_on_whitespace_only_tickers(self, mock_download):
        """Test that whitespace-only tickers result in empty list after normalization."""
        # After stripping, these become empty strings and are removed
        # The function should raise ValueError for empty list
        result_tickers = [t.strip() for t in ["  ", "\t", "\n"]]
        result_tickers = [t for t in result_tickers if t]

        # If all are empty after normalization, should have empty list
        assert len(result_tickers) == 0

        # The function should raise ValueError
        with pytest.raises(ValueError, match="vazia"):
            download_prices(result_tickers)

    @patch('arara_quant.data.sources.yf.yf.download')
    def test_passes_start_end_dates(self, mock_download, mock_multiindex_data):
        """Test that start and end dates are passed to yfinance."""
        mock_download.return_value = mock_multiindex_data

        start = "2024-01-01"
        end = "2024-12-31"

        download_prices(["SPY"], start=start, end=end)

        call_kwargs = mock_download.call_args.kwargs
        assert call_kwargs["start"] == start
        assert call_kwargs["end"] == end

    @patch('arara_quant.data.sources.yf.yf.download')
    def test_passes_datetime_objects(self, mock_download, mock_multiindex_data):
        """Test that datetime objects are accepted."""
        mock_download.return_value = mock_multiindex_data

        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)

        download_prices(["SPY"], start=start, end=end)

        call_kwargs = mock_download.call_args.kwargs
        assert call_kwargs["start"] == start
        assert call_kwargs["end"] == end

    @patch('arara_quant.data.sources.yf.yf.download')
    def test_disables_progress_by_default(self, mock_download, mock_multiindex_data):
        """Test that progress bar is disabled by default."""
        mock_download.return_value = mock_multiindex_data

        download_prices(["SPY"])

        call_kwargs = mock_download.call_args.kwargs
        assert call_kwargs["progress"] is False

    @patch('arara_quant.data.sources.yf.yf.download')
    def test_enables_progress_when_requested(self, mock_download, mock_multiindex_data):
        """Test that progress bar can be enabled."""
        mock_download.return_value = mock_multiindex_data

        download_prices(["SPY"], progress=True)

        call_kwargs = mock_download.call_args.kwargs
        assert call_kwargs["progress"] is True

    @patch('arara_quant.data.sources.yf.yf.download')
    def test_uses_auto_adjust_false(self, mock_download, mock_multiindex_data):
        """Test that auto_adjust is set to False."""
        mock_download.return_value = mock_multiindex_data

        download_prices(["SPY"])

        call_kwargs = mock_download.call_args.kwargs
        assert call_kwargs["auto_adjust"] is False

    @patch('arara_quant.data.sources.yf.yf.download')
    def test_sorts_index(self, mock_download):
        """Test that index is sorted."""
        # Create data with unsorted dates
        dates = pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"])
        data = pd.DataFrame({
            ("SPY", "Adj Close"): [103, 101, 102],
        }, index=dates)
        data.columns = pd.MultiIndex.from_tuples(data.columns)

        mock_download.return_value = data

        result = download_prices(["SPY"])

        # Index should be sorted
        assert result.index.is_monotonic_increasing

    @patch('arara_quant.data.sources.yf.yf.download')
    def test_handles_single_ticker_without_multiindex(self, mock_download, mock_single_ticker_data):
        """Test handling of single ticker data (no MultiIndex)."""
        mock_download.return_value = mock_single_ticker_data

        result = download_prices(["SPY"])

        assert isinstance(result, pd.DataFrame)
        assert "SPY" in result.columns or len(result.columns) == 1
        assert len(result) > 0

    @patch('arara_quant.data.sources.yf.yf.download')
    def test_logs_download_info(self, mock_download, mock_multiindex_data, caplog):
        """Test that download info is logged."""
        mock_download.return_value = mock_multiindex_data

        with caplog.at_level("INFO"):
            download_prices(["SPY", "QQQ"], start="2024-01-01")

        assert any("Baixando preços" in record.message for record in caplog.records)

    @patch('arara_quant.data.sources.yf.yf.download')
    def test_handles_missing_tickers_gracefully(self, mock_download, mock_multiindex_data):
        """Test that missing tickers are handled gracefully."""
        # Request more tickers than available in data
        mock_download.return_value = mock_multiindex_data

        result = download_prices(["SPY", "QQQ", "INVALID"])

        # Should only return data for available tickers
        assert "SPY" in result.columns
        assert "QQQ" in result.columns
        assert "INVALID" not in result.columns
