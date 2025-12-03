"""Data downloader for cryptocurrency market data using CCXT."""

import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import ccxt

logger = logging.getLogger(__name__)


class DataDownloadError(Exception):
    """Exception raised for data download errors."""
    pass


class DataDownloader:
    """Downloads OHLCV (Open, High, Low, Close, Volume) data from cryptocurrency exchanges."""

    # Rate limiting constants (milliseconds)
    DEFAULT_RATE_LIMIT = 1000  # 1 second between requests
    RETRY_ATTEMPTS = 3
    RETRY_DELAY = 2  # seconds

    def __init__(
        self,
        exchange_name: str = "binance",
        rate_limit_ms: int = DEFAULT_RATE_LIMIT,
        data_dir: str = "data",
    ):
        """
        Initialize DataDownloader with exchange connection.

        Args:
            exchange_name: Name of the exchange (e.g., 'binance', 'bybit')
            rate_limit_ms: Rate limit in milliseconds between API calls
            data_dir: Directory to save downloaded data

        Raises:
            DataDownloadError: If exchange is not supported or connection fails
        """
        self.exchange_name = exchange_name.lower()
        self.rate_limit_ms = rate_limit_ms
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize exchange
        try:
            exchange_class = getattr(ccxt, self.exchange_name)
            self.exchange = exchange_class({"enableRateLimit": True})
            logger.info(f"Connected to {self.exchange_name} exchange")
        except AttributeError:
            raise DataDownloadError(
                f"Exchange '{self.exchange_name}' not supported. "
                f"Available exchanges: {', '.join(ccxt.exchanges)}"
            )
        except Exception as e:
            raise DataDownloadError(f"Failed to initialize exchange: {str(e)}")

    def download_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Download OHLCV data from exchange.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Candlestick timeframe (e.g., '1m', '5m', '1h', '1d')
            start_date: Start date in format 'YYYY-MM-DD' (default: 1 year ago)
            end_date: End date in format 'YYYY-MM-DD' (default: today)
            limit: Maximum candles per request (default: 1000)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume

        Raises:
            DataDownloadError: If download fails or symbol is invalid
        """
        # Validate symbol
        if not self._validate_symbol(symbol):
            raise DataDownloadError(f"Symbol '{symbol}' not available on {self.exchange_name}")

        # Parse dates
        if start_date is None:
            start_dt = datetime.now() - timedelta(days=365)
        else:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")

        if end_date is None:
            end_dt = datetime.now()
        else:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        if start_dt >= end_dt:
            raise DataDownloadError("start_date must be before end_date")

        logger.info(
            f"Downloading {symbol} {timeframe} data from {start_dt.date()} to {end_dt.date()}"
        )

        all_candles = []
        current_dt = start_dt

        while current_dt < end_dt:
            try:
                # Convert to milliseconds for CCXT
                since = int(current_dt.timestamp() * 1000)

                # Fetch with retry logic
                candles = self._fetch_with_retry(symbol, timeframe, since, limit)

                if not candles:
                    logger.warning(f"No data returned for {symbol} at {current_dt}")
                    break

                all_candles.extend(candles)

                # Move to next batch
                last_candle_time = candles[-1][0]
                current_dt = datetime.fromtimestamp(last_candle_time / 1000)

                # Stop if we've reached end date
                if current_dt >= end_dt:
                    break

                # Rate limiting
                time.sleep(self.rate_limit_ms / 1000.0)

            except Exception as e:
                raise DataDownloadError(f"Failed to download {symbol} data: {str(e)}")

        if not all_candles:
            raise DataDownloadError(f"No data downloaded for {symbol}")

        # Convert to DataFrame
        df = pd.DataFrame(
            all_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )

        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        # Filter to requested date range
        df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)]

        # Remove duplicates
        df = df.drop_duplicates(subset=["timestamp"], keep="first")

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info(f"Downloaded {len(df)} candles for {symbol}")

        return df

    def save_to_csv(
        self,
        data: pd.DataFrame,
        symbol: str,
        timeframe: str = "1h",
        filename: Optional[str] = None,
    ) -> Path:
        """
        Save downloaded data to CSV file.

        Args:
            data: DataFrame with OHLCV data
            symbol: Trading pair symbol (used in filename if not specified)
            timeframe: Candlestick timeframe (used in filename if not specified)
            filename: Custom filename (default: {symbol}_{timeframe}.csv)

        Returns:
            Path to saved CSV file

        Raises:
            DataDownloadError: If save fails
        """
        try:
            if filename is None:
                # Create filename from symbol and timeframe
                safe_symbol = symbol.replace("/", "_")
                filename = f"{safe_symbol}_{timeframe}.csv"

            filepath = self.data_dir / filename

            # Save to CSV
            data.to_csv(filepath, index=False)
            logger.info(f"Saved data to {filepath}")

            return filepath

        except Exception as e:
            raise DataDownloadError(f"Failed to save data to CSV: {str(e)}")

    def _validate_symbol(self, symbol: str) -> bool:
        """
        Validate that symbol is available on exchange.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')

        Returns:
            True if symbol is available, False otherwise
        """
        try:
            if not self.exchange.symbols:
                self.exchange.load_markets()
            return symbol in self.exchange.symbols
        except Exception as e:
            logger.warning(f"Failed to validate symbol: {str(e)}")
            return False

    def _fetch_with_retry(
        self,
        symbol: str,
        timeframe: str,
        since: int,
        limit: int,
    ) -> List[List[Any]]:
        """
        Fetch OHLCV data with retry logic.

        Args:
            symbol: Trading pair symbol
            timeframe: Candlestick timeframe
            since: Start time in milliseconds
            limit: Maximum candles to fetch

        Returns:
            List of OHLCV candles

        Raises:
            DataDownloadError: If all retry attempts fail
        """
        last_error = None

        for attempt in range(self.RETRY_ATTEMPTS):
            try:
                candles = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
                return candles
            except ccxt.RateLimitExceeded:
                wait_time = self.RETRY_DELAY * (2 ** attempt)
                logger.warning(
                    f"Rate limit exceeded. Waiting {wait_time}s before retry "
                    f"(attempt {attempt + 1}/{self.RETRY_ATTEMPTS})"
                )
                time.sleep(wait_time)
                last_error = "Rate limit exceeded"
            except ccxt.NetworkError as e:
                wait_time = self.RETRY_DELAY * (2 ** attempt)
                logger.warning(
                    f"Network error: {str(e)}. Waiting {wait_time}s before retry "
                    f"(attempt {attempt + 1}/{self.RETRY_ATTEMPTS})"
                )
                time.sleep(wait_time)
                last_error = str(e)
            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error: {str(e)}")
                raise DataDownloadError(f"Exchange error: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                raise DataDownloadError(f"Unexpected error: {str(e)}")

        raise DataDownloadError(
            f"Failed to fetch data after {self.RETRY_ATTEMPTS} attempts. "
            f"Last error: {last_error}"
        )

    def get_available_symbols(self) -> List[str]:
        """
        Get list of available trading symbols on exchange.

        Returns:
            List of available symbols

        Raises:
            DataDownloadError: If unable to fetch symbols
        """
        try:
            if not self.exchange.symbols:
                self.exchange.load_markets()
            return self.exchange.symbols
        except Exception as e:
            raise DataDownloadError(f"Failed to fetch available symbols: {str(e)}")

    def get_available_timeframes(self) -> List[str]:
        """
        Get list of available timeframes on exchange.

        Returns:
            List of available timeframes

        Raises:
            DataDownloadError: If unable to fetch timeframes
        """
        try:
            if not self.exchange.timeframes:
                self.exchange.load_markets()
            return list(self.exchange.timeframes.keys()) if self.exchange.timeframes else []
        except Exception as e:
            raise DataDownloadError(f"Failed to fetch available timeframes: {str(e)}")
