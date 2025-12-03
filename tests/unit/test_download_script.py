"""Unit tests for download_script CLI module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from argparse import Namespace
import tempfile
from pathlib import Path

from src.data.download_script import download_command, _normalize_symbol
from src.data.data_downloader import DataDownloadError
from io import StringIO


class TestSymbolNormalization:
    """Test symbol normalization functionality."""

    def test_normalize_coin_name_lowercase(self):
        """Test normalization of lowercase coin name."""
        result = _normalize_symbol("btc")
        assert result == "BTC/USDT"

    def test_normalize_coin_name_uppercase(self):
        """Test normalization of uppercase coin name."""
        result = _normalize_symbol("BTC")
        assert result == "BTC/USDT"

    def test_normalize_coin_name_mixed_case(self):
        """Test normalization of mixed case coin name."""
        result = _normalize_symbol("EtH")
        assert result == "ETH/USDT"

    def test_normalize_already_formatted_pair(self):
        """Test that already formatted pairs are returned as-is."""
        result = _normalize_symbol("BTC/USDT")
        assert result == "BTC/USDT"

    def test_normalize_different_quote_currency(self):
        """Test normalization with different quote currency."""
        result = _normalize_symbol("BTC", quote_currency="BUSD")
        assert result == "BTC/BUSD"

    def test_normalize_multiple_coins(self):
        """Test normalization of multiple coin names."""
        coins = ["BTC", "ETH", "ADA"]
        normalized = [_normalize_symbol(coin) for coin in coins]
        assert normalized == ["BTC/USDT", "ETH/USDT", "ADA/USDT"]

    def test_normalize_mixed_formats(self):
        """Test normalization of mixed coin names and pairs."""
        symbols = ["BTC", "ETH/USDT", "ADA"]
        normalized = [_normalize_symbol(sym) for sym in symbols]
        assert normalized == ["BTC/USDT", "ETH/USDT", "ADA/USDT"]


class TestDownloadCommand:
    """Test download_command CLI handler."""

    def test_download_command_single_symbol_single_timeframe(self):
        """Test download command with single coin name and timeframe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = Namespace(
                symbols=["BTC"],
                timeframes=["1h"],
                exchange="binance",
                market_type="future",
                start_date=None,
                end_date=None,
                data_dir=tmpdir,
            )

            with patch("src.data.download_script.DataDownloader") as mock_downloader_class:
                mock_downloader = MagicMock()
                mock_downloader_class.return_value = mock_downloader

                # Mock the download and save methods
                import pandas as pd
                from datetime import datetime
                mock_df = pd.DataFrame({
                    "timestamp": [datetime(2021, 1, 1)],
                    "open": [29000],
                    "high": [30000],
                    "low": [28000],
                    "close": [29500],
                    "volume": [100],
                })
                mock_downloader.download_ohlcv.return_value = mock_df
                mock_downloader.save_to_csv.return_value = Path(tmpdir) / "BTC_USDT_1h_future.csv"

                # Execute command
                download_command(args)

                # Verify downloader was initialized correctly
                mock_downloader_class.assert_called_once_with(
                    exchange_name="binance",
                    market_type="future",
                    data_dir=tmpdir,
                )

                # Verify download was called with normalized symbol (BTC -> BTC/USDT)
                call_args = mock_downloader.download_ohlcv.call_args
                assert call_args[1]["symbol"] == "BTC/USDT"
                assert call_args[1]["timeframe"] == "1h"
                assert call_args[1]["start_date"] is None
                assert call_args[1]["end_date"] is not None  # Should be today's date

                # Verify save was called
                mock_downloader.save_to_csv.assert_called_once()

    def test_download_command_multiple_symbols_multiple_timeframes(self):
        """Test download command with multiple coin names and timeframes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = Namespace(
                symbols=["BTC", "ETH"],
                timeframes=["1h", "4h"],
                exchange="binance",
                market_type="future",
                start_date=None,
                end_date=None,
                data_dir=tmpdir,
            )

            with patch("src.data.download_script.DataDownloader") as mock_downloader_class:
                mock_downloader = MagicMock()
                mock_downloader_class.return_value = mock_downloader

                # Mock the download and save methods
                import pandas as pd
                from datetime import datetime
                mock_df = pd.DataFrame({
                    "timestamp": [datetime(2021, 1, 1)],
                    "open": [29000],
                    "high": [30000],
                    "low": [28000],
                    "close": [29500],
                    "volume": [100],
                })
                mock_downloader.download_ohlcv.return_value = mock_df
                mock_downloader.save_to_csv.return_value = Path(tmpdir) / "data.csv"

                # Execute command
                download_command(args)

                # Verify download was called 4 times (2 symbols × 2 timeframes)
                assert mock_downloader.download_ohlcv.call_count == 4
                assert mock_downloader.save_to_csv.call_count == 4
                
                # Verify symbols were normalized
                calls = mock_downloader.download_ohlcv.call_args_list
                symbols_used = {call[1]["symbol"] for call in calls}
                assert symbols_used == {"BTC/USDT", "ETH/USDT"}

    def test_download_command_with_date_range(self):
        """Test download command with specific date range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = Namespace(
                symbols=["BTC"],
                timeframes=["1h"],
                exchange="binance",
                market_type="spot",
                start_date="2023-01-01",
                end_date="2023-12-31",
                data_dir=tmpdir,
            )

            with patch("src.data.download_script.DataDownloader") as mock_downloader_class:
                mock_downloader = MagicMock()
                mock_downloader_class.return_value = mock_downloader

                # Mock the download and save methods
                import pandas as pd
                from datetime import datetime
                mock_df = pd.DataFrame({
                    "timestamp": [datetime(2023, 1, 1)],
                    "open": [29000],
                    "high": [30000],
                    "low": [28000],
                    "close": [29500],
                    "volume": [100],
                })
                mock_downloader.download_ohlcv.return_value = mock_df
                mock_downloader.save_to_csv.return_value = Path(tmpdir) / "data.csv"

                # Execute command
                download_command(args)

                # Verify download was called with correct date range and normalized symbol
                call_args = mock_downloader.download_ohlcv.call_args
                assert call_args[1]["symbol"] == "BTC/USDT"
                assert call_args[1]["timeframe"] == "1h"
                assert call_args[1]["start_date"] == "2023-01-01"
                assert call_args[1]["end_date"] == "2023-12-31"

    def test_download_command_without_start_date(self):
        """Test download command without start date uses earliest available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = Namespace(
                symbols=["BTC"],
                timeframes=["1h"],
                exchange="binance",
                market_type="future",
                start_date=None,
                end_date=None,
                data_dir=tmpdir,
            )

            with patch("src.data.download_script.DataDownloader") as mock_downloader_class:
                mock_downloader = MagicMock()
                mock_downloader_class.return_value = mock_downloader

                # Mock the download and save methods
                import pandas as pd
                from datetime import datetime
                mock_df = pd.DataFrame({
                    "timestamp": [datetime(2021, 1, 1)],
                    "open": [29000],
                    "high": [30000],
                    "low": [28000],
                    "close": [29500],
                    "volume": [100],
                })
                mock_downloader.download_ohlcv.return_value = mock_df
                mock_downloader.save_to_csv.return_value = Path(tmpdir) / "data.csv"

                # Execute command
                download_command(args)

                # Verify download was called with None for start_date (earliest available)
                call_args = mock_downloader.download_ohlcv.call_args
                assert call_args[1]["start_date"] is None
                assert call_args[1]["end_date"] is not None  # Should have today's date

    def test_download_command_handles_download_error(self):
        """Test download command handles DataDownloadError gracefully."""
        args = Namespace(
            symbols=["BTC"],
            timeframes=["1h"],
            exchange="binance",
            market_type="future",
            start_date=None,
            end_date=None,
            data_dir="data",
        )

        with patch("src.data.download_script.DataDownloader") as mock_downloader_class:
            mock_downloader = MagicMock()
            mock_downloader_class.return_value = mock_downloader

            # Mock download to raise error
            mock_downloader.download_ohlcv.side_effect = DataDownloadError("Network error")

            # Execute command - should exit with error code 1 when there are failures
            with pytest.raises(SystemExit) as exc_info:
                download_command(args)
            
            assert exc_info.value.code == 1

            # Verify download was attempted
            mock_downloader.download_ohlcv.assert_called_once()

    def test_download_command_initializer_error(self):
        """Test download command handles initialization error."""
        args = Namespace(
            symbols=["BTC/USDT"],
            timeframes=["1h"],
            exchange="invalid_exchange",
            market_type="future",
            start_date=None,
            end_date=None,
            data_dir="data",
        )

        with patch("src.data.download_script.DataDownloader") as mock_downloader_class:
            # Mock initialization to raise error
            mock_downloader_class.side_effect = DataDownloadError("Exchange not supported")

            # Execute command - should exit with error
            with pytest.raises(SystemExit):
                download_command(args)

    def test_download_command_with_bybit_exchange(self):
        """Test download command with Bybit exchange."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = Namespace(
                symbols=["BTC"],
                timeframes=["1h"],
                exchange="bybit",
                market_type="future",
                start_date=None,
                end_date=None,
                data_dir=tmpdir,
            )

            with patch("src.data.download_script.DataDownloader") as mock_downloader_class:
                mock_downloader = MagicMock()
                mock_downloader_class.return_value = mock_downloader

                # Mock the download and save methods
                import pandas as pd
                from datetime import datetime
                mock_df = pd.DataFrame({
                    "timestamp": [datetime(2021, 1, 1)],
                    "open": [29000],
                    "high": [30000],
                    "low": [28000],
                    "close": [29500],
                    "volume": [100],
                })
                mock_downloader.download_ohlcv.return_value = mock_df
                mock_downloader.save_to_csv.return_value = Path(tmpdir) / "data.csv"

                # Execute command
                download_command(args)

                # Verify downloader was initialized with bybit
                mock_downloader_class.assert_called_once_with(
                    exchange_name="bybit",
                    market_type="future",
                    data_dir=tmpdir,
                )
