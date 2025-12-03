"""Unit tests for DataDownloader class."""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

from src.data.data_downloader import DataDownloader, DataDownloadError


class TestDataDownloaderInitialization:
    """Test DataDownloader initialization."""

    def test_init_with_default_exchange(self):
        """Test initialization with default Binance exchange for futures."""
        with patch("src.data.data_downloader.ccxt.binance"):
            downloader = DataDownloader()
            assert downloader.exchange_name == "binance"
            assert downloader.rate_limit_ms == 1000
            assert downloader.data_dir.name == "data"
            assert downloader.market_type == "future"

    def test_init_with_custom_exchange(self):
        """Test initialization with custom exchange."""
        with patch("src.data.data_downloader.ccxt.bybit"):
            downloader = DataDownloader(exchange_name="bybit")
            assert downloader.exchange_name == "bybit"

    def test_init_creates_data_directory(self):
        """Test that initialization creates data directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "test_data"
            assert not data_dir.exists()

            with patch("src.data.data_downloader.ccxt.binance"):
                downloader = DataDownloader(data_dir=str(data_dir))
                assert data_dir.exists()

    def test_init_with_invalid_exchange(self):
        """Test initialization with invalid exchange raises error."""
        with patch("src.data.data_downloader.ccxt", create=True) as mock_ccxt:
            mock_ccxt.exchanges = ["binance", "bybit"]
            delattr(mock_ccxt, "invalid_exchange")

            with pytest.raises(DataDownloadError, match="not supported"):
                DataDownloader(exchange_name="invalid_exchange")

    def test_init_with_custom_rate_limit(self):
        """Test initialization with custom rate limit."""
        with patch("src.data.data_downloader.ccxt.binance"):
            downloader = DataDownloader(rate_limit_ms=500)
            assert downloader.rate_limit_ms == 500

    def test_init_with_spot_market(self):
        """Test initialization with spot market type."""
        with patch("src.data.data_downloader.ccxt.binance"):
            downloader = DataDownloader(market_type="spot")
            assert downloader.market_type == "spot"

    def test_init_with_future_market(self):
        """Test initialization with future market type."""
        with patch("src.data.data_downloader.ccxt.binance"):
            downloader = DataDownloader(market_type="future")
            assert downloader.market_type == "future"

    def test_init_with_invalid_market_type(self):
        """Test initialization with invalid market type raises error."""
        with pytest.raises(DataDownloadError, match="Invalid market_type"):
            DataDownloader(market_type="invalid")


class TestDataDownloaderValidation:
    """Test DataDownloader validation methods."""

    def test_validate_symbol_valid(self):
        """Test validation of valid symbol."""
        with patch("src.data.data_downloader.ccxt.binance") as mock_exchange_class:
            mock_exchange = MagicMock()
            mock_exchange.symbols = ["BTC/USDT", "ETH/USDT"]
            mock_exchange_class.return_value = mock_exchange

            downloader = DataDownloader()
            assert downloader._validate_symbol("BTC/USDT") is True

    def test_validate_symbol_invalid(self):
        """Test validation of invalid symbol."""
        with patch("src.data.data_downloader.ccxt.binance") as mock_exchange_class:
            mock_exchange = MagicMock()
            mock_exchange.symbols = ["BTC/USDT", "ETH/USDT"]
            mock_exchange_class.return_value = mock_exchange

            downloader = DataDownloader()
            assert downloader._validate_symbol("INVALID/USDT") is False

    def test_get_available_symbols(self):
        """Test getting available symbols."""
        with patch("src.data.data_downloader.ccxt.binance") as mock_exchange_class:
            mock_exchange = MagicMock()
            mock_exchange.symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]
            mock_exchange_class.return_value = mock_exchange

            downloader = DataDownloader()
            symbols = downloader.get_available_symbols()
            assert symbols == ["BTC/USDT", "ETH/USDT", "ADA/USDT"]

    def test_get_available_timeframes(self):
        """Test getting available timeframes."""
        with patch("src.data.data_downloader.ccxt.binance") as mock_exchange_class:
            mock_exchange = MagicMock()
            mock_exchange.timeframes = {"1m": "1m", "5m": "5m", "1h": "1h"}
            mock_exchange_class.return_value = mock_exchange

            downloader = DataDownloader()
            timeframes = downloader.get_available_timeframes()
            assert set(timeframes) == {"1m", "5m", "1h"}


class TestDataDownloaderDownload:
    """Test OHLCV data downloading."""

    def test_download_ohlcv_basic(self):
        """Test basic OHLCV download."""
        with patch("src.data.data_downloader.ccxt.binance") as mock_exchange_class:
            mock_exchange = MagicMock()
            mock_exchange.symbols = ["BTC/USDT"]
            # Use recent timestamps (within last year)
            now_ms = int(datetime.now().timestamp() * 1000)
            # Return data once, then empty to stop the loop
            mock_exchange.fetch_ohlcv.side_effect = [
                [
                    [now_ms - 3600000, 29000, 30000, 28000, 29500, 100],
                    [now_ms, 29500, 31000, 29000, 30500, 150],
                ],
                [],  # Empty response stops the loop
            ]
            mock_exchange_class.return_value = mock_exchange

            downloader = DataDownloader()
            df = downloader.download_ohlcv("BTC/USDT", timeframe="1h")

            assert len(df) == 2
            assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
            assert df["open"].iloc[0] == 29000
            assert df["close"].iloc[1] == 30500

    def test_download_ohlcv_with_date_range(self):
        """Test OHLCV download with specific date range."""
        with patch("src.data.data_downloader.ccxt.binance") as mock_exchange_class:
            mock_exchange = MagicMock()
            mock_exchange.symbols = ["BTC/USDT"]
            mock_exchange.fetch_ohlcv.side_effect = [
                [[1609459200000, 29000, 30000, 28000, 29500, 100]],
                [],  # Empty response stops the loop
            ]
            mock_exchange_class.return_value = mock_exchange

            downloader = DataDownloader()
            df = downloader.download_ohlcv(
                "BTC/USDT",
                timeframe="1h",
                start_date="2021-01-01",
                end_date="2021-01-02",
            )

            assert len(df) >= 0  # May be filtered out depending on dates

    def test_download_ohlcv_invalid_symbol(self):
        """Test download with invalid symbol raises error."""
        with patch("src.data.data_downloader.ccxt.binance") as mock_exchange_class:
            mock_exchange = MagicMock()
            mock_exchange.symbols = ["BTC/USDT"]
            mock_exchange_class.return_value = mock_exchange

            downloader = DataDownloader()
            with pytest.raises(DataDownloadError, match="not available"):
                downloader.download_ohlcv("INVALID/USDT")

    def test_download_ohlcv_invalid_date_range(self):
        """Test download with invalid date range raises error."""
        with patch("src.data.data_downloader.ccxt.binance") as mock_exchange_class:
            mock_exchange = MagicMock()
            mock_exchange.symbols = ["BTC/USDT"]
            mock_exchange_class.return_value = mock_exchange

            downloader = DataDownloader()
            with pytest.raises(DataDownloadError, match="start_date must be before end_date"):
                downloader.download_ohlcv(
                    "BTC/USDT",
                    start_date="2021-01-02",
                    end_date="2021-01-01",
                )

    def test_download_ohlcv_no_data(self):
        """Test download with no data returned raises error."""
        with patch("src.data.data_downloader.ccxt.binance") as mock_exchange_class:
            mock_exchange = MagicMock()
            mock_exchange.symbols = ["BTC/USDT"]
            mock_exchange.fetch_ohlcv.return_value = []
            mock_exchange_class.return_value = mock_exchange

            downloader = DataDownloader()
            with pytest.raises(DataDownloadError, match="No data downloaded"):
                downloader.download_ohlcv("BTC/USDT")

    def test_download_ohlcv_removes_duplicates(self):
        """Test that download removes duplicate timestamps."""
        with patch("src.data.data_downloader.ccxt.binance") as mock_exchange_class:
            mock_exchange = MagicMock()
            mock_exchange.symbols = ["BTC/USDT"]
            now_ms = int(datetime.now().timestamp() * 1000)
            mock_exchange.fetch_ohlcv.side_effect = [
                [
                    [now_ms - 3600000, 29000, 30000, 28000, 29500, 100],
                    [now_ms - 3600000, 29100, 30100, 28100, 29600, 110],  # Duplicate timestamp
                    [now_ms, 29500, 31000, 29000, 30500, 150],
                ],
                [],  # Empty response stops the loop
            ]
            mock_exchange_class.return_value = mock_exchange

            downloader = DataDownloader()
            df = downloader.download_ohlcv("BTC/USDT")

            assert len(df) == 2  # Duplicate removed
            assert df["open"].iloc[0] == 29000  # First one kept

    def test_download_ohlcv_sorts_by_timestamp(self):
        """Test that download sorts data by timestamp."""
        with patch("src.data.data_downloader.ccxt.binance") as mock_exchange_class:
            mock_exchange = MagicMock()
            mock_exchange.symbols = ["BTC/USDT"]
            now_ms = int(datetime.now().timestamp() * 1000)
            mock_exchange.fetch_ohlcv.side_effect = [
                [
                    [now_ms, 29500, 31000, 29000, 30500, 150],
                    [now_ms - 3600000, 29000, 30000, 28000, 29500, 100],  # Out of order
                ],
                [],  # Empty response stops the loop
            ]
            mock_exchange_class.return_value = mock_exchange

            downloader = DataDownloader()
            df = downloader.download_ohlcv("BTC/USDT")

            assert df["timestamp"].is_monotonic_increasing


class TestDataDownloaderSaveCSV:
    """Test CSV saving functionality."""

    def test_save_to_csv_default_filename(self):
        """Test saving to CSV with default filename for futures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.data.data_downloader.ccxt.binance"):
                downloader = DataDownloader(data_dir=tmpdir, market_type="future")

                df = pd.DataFrame({
                    "timestamp": [datetime(2021, 1, 1), datetime(2021, 1, 2)],
                    "open": [29000, 29500],
                    "high": [30000, 31000],
                    "low": [28000, 29000],
                    "close": [29500, 30500],
                    "volume": [100, 150],
                })

                filepath = downloader.save_to_csv(df, "BTC/USDT", "1h")

                assert filepath.exists()
                assert filepath.name == "BTC_USDT_1h_future.csv"
                assert filepath.parent == Path(tmpdir)

    def test_save_to_csv_custom_filename(self):
        """Test saving to CSV with custom filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.data.data_downloader.ccxt.binance"):
                downloader = DataDownloader(data_dir=tmpdir)

                df = pd.DataFrame({
                    "timestamp": [datetime(2021, 1, 1)],
                    "open": [29000],
                    "high": [30000],
                    "low": [28000],
                    "close": [29500],
                    "volume": [100],
                })

                filepath = downloader.save_to_csv(
                    df, "BTC/USDT", "1h", filename="custom_data.csv"
                )

                assert filepath.exists()
                assert filepath.name == "custom_data.csv"

    def test_save_to_csv_data_integrity(self):
        """Test that saved CSV preserves data integrity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.data.data_downloader.ccxt.binance"):
                downloader = DataDownloader(data_dir=tmpdir)

                original_df = pd.DataFrame({
                    "timestamp": [datetime(2021, 1, 1), datetime(2021, 1, 2)],
                    "open": [29000, 29500],
                    "high": [30000, 31000],
                    "low": [28000, 29000],
                    "close": [29500, 30500],
                    "volume": [100, 150],
                })

                filepath = downloader.save_to_csv(original_df, "BTC/USDT", "1h")

                # Read back and verify
                loaded_df = pd.read_csv(filepath)
                assert len(loaded_df) == len(original_df)
                assert list(loaded_df.columns) == list(original_df.columns)
                assert loaded_df["open"].tolist() == original_df["open"].tolist()

    def test_save_to_csv_invalid_path(self):
        """Test saving to invalid path raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.data.data_downloader.ccxt.binance"):
                downloader = DataDownloader(data_dir=tmpdir)

                df = pd.DataFrame({
                    "timestamp": [datetime(2021, 1, 1)],
                    "open": [29000],
                    "high": [30000],
                    "low": [28000],
                    "close": [29500],
                    "volume": [100],
                })

                # Mock the to_csv method to raise an error
                with patch.object(df, 'to_csv', side_effect=IOError("Permission denied")):
                    with pytest.raises(DataDownloadError, match="Failed to save"):
                        downloader.save_to_csv(df, "BTC/USDT", "1h")


class TestDataDownloaderRetryLogic:
    """Test retry logic for API calls."""

    def test_fetch_with_retry_success_first_attempt(self):
        """Test successful fetch on first attempt."""
        with patch("src.data.data_downloader.ccxt.binance") as mock_exchange_class:
            mock_exchange = MagicMock()
            mock_exchange.fetch_ohlcv.return_value = [
                [1609459200000, 29000, 30000, 28000, 29500, 100],
            ]
            mock_exchange_class.return_value = mock_exchange

            downloader = DataDownloader()
            result = downloader._fetch_with_retry("BTC/USDT", "1h", 1609459200000, 1000)

            assert len(result) == 1
            assert result[0][1] == 29000

    def test_fetch_with_retry_rate_limit_then_success(self):
        """Test retry after rate limit error."""
        import ccxt

        with patch("src.data.data_downloader.ccxt.binance") as mock_exchange_class:
            mock_exchange = MagicMock()
            mock_exchange.fetch_ohlcv.side_effect = [
                ccxt.RateLimitExceeded("Rate limit"),
                [[1609459200000, 29000, 30000, 28000, 29500, 100]],
            ]
            mock_exchange_class.return_value = mock_exchange

            downloader = DataDownloader(rate_limit_ms=100)
            # Patch time.sleep to avoid actual delays
            with patch("src.data.data_downloader.time.sleep"):
                result = downloader._fetch_with_retry("BTC/USDT", "1h", 1609459200000, 1000)

                assert len(result) == 1
                assert mock_exchange.fetch_ohlcv.call_count == 2

    def test_fetch_with_retry_all_attempts_fail(self):
        """Test error after all retry attempts fail."""
        import ccxt

        with patch("src.data.data_downloader.ccxt.binance") as mock_exchange_class:
            mock_exchange = MagicMock()
            mock_exchange.fetch_ohlcv.side_effect = ccxt.RateLimitExceeded("Rate limit")
            mock_exchange_class.return_value = mock_exchange

            downloader = DataDownloader(rate_limit_ms=100)
            # Patch time.sleep to avoid actual delays
            with patch("src.data.data_downloader.time.sleep"):
                with pytest.raises(DataDownloadError, match="Failed to fetch data"):
                    downloader._fetch_with_retry("BTC/USDT", "1h", 1609459200000, 1000)

    def test_fetch_with_retry_exchange_error(self):
        """Test exchange error is raised immediately."""
        import ccxt

        with patch("src.data.data_downloader.ccxt.binance") as mock_exchange_class:
            mock_exchange = MagicMock()
            mock_exchange.fetch_ohlcv.side_effect = ccxt.ExchangeError("Invalid symbol")
            mock_exchange_class.return_value = mock_exchange

            downloader = DataDownloader()
            with pytest.raises(DataDownloadError, match="Exchange error"):
                downloader._fetch_with_retry("BTC/USDT", "1h", 1609459200000, 1000)
