"""Data downloader for cryptocurrency market data using CCXT."""

import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import ccxt
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Timeframe to minutes mapping for estimating total candles
TIMEFRAME_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480, "12h": 720,
    "1d": 1440, "3d": 4320, "1w": 10080, "1M": 43200,
}


class DataDownloadError(Exception):
    """Exception raised for data download errors."""
    pass


class DataDownloader:
    """Downloads OHLCV (Open, High, Low, Close, Volume) data from cryptocurrency futures exchanges."""

    # Rate limiting constants (milliseconds)
    DEFAULT_RATE_LIMIT = 1000  # 1 second between requests
    RETRY_ATTEMPTS = 3
    RETRY_DELAY = 2  # seconds

    def __init__(
        self,
        exchange_name: str = "binance",
        rate_limit_ms: int = DEFAULT_RATE_LIMIT,
        data_dir: str = "data",
        market_type: str = "future",
    ):
        """
        Initialize DataDownloader with exchange connection for futures trading.

        Args:
            exchange_name: Name of the exchange (e.g., 'binance', 'bybit')
            rate_limit_ms: Rate limit in milliseconds between API calls
            data_dir: Directory to save downloaded data
            market_type: Market type - 'future' for perpetual/futures, 'spot' for spot trading

        Raises:
            DataDownloadError: If exchange is not supported or connection fails
        """
        self.exchange_name = exchange_name.lower()
        self.rate_limit_ms = rate_limit_ms
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.market_type = market_type.lower()

        if self.market_type not in ["future", "spot"]:
            raise DataDownloadError(f"Invalid market_type '{market_type}'. Must be 'future' or 'spot'")

        # Initialize exchange
        try:
            exchange_class = getattr(ccxt, self.exchange_name)
            exchange_config = {"enableRateLimit": True}
            
            # Configure for futures if needed
            if self.market_type == "future":
                exchange_config["options"] = {"defaultType": "future"}
            
            self.exchange = exchange_class(exchange_config)
            logger.info(f"Connected to {self.exchange_name} exchange ({self.market_type} market)")
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
        progress_callback: Optional[callable] = None,
    ) -> pd.DataFrame:
        """
        Download OHLCV data from futures exchange.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT:USDT' for futures, 'BTC/USDT' for spot)
            timeframe: Candlestick timeframe (e.g., '1m', '5m', '1h', '1d')
            start_date: Start date in format 'YYYY-MM-DD' (default: None = earliest available)
            end_date: End date in format 'YYYY-MM-DD' (default: today)
            limit: Maximum candles per request (default: 1000)
            progress_callback: Optional callback function(downloaded_count) for progress updates

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume

        Raises:
            DataDownloadError: If download fails or symbol is invalid
        """
        # Validate symbol
        if not self._validate_symbol(symbol):
            raise DataDownloadError(
                f"Symbol '{symbol}' not available on {self.exchange_name} ({self.market_type} market)"
            )

        # Parse dates
        # If start_date is None, we'll fetch from a very old date to get all available data
        # Most exchanges have data going back to 2017 at the earliest
        if start_date is None:
            start_dt = datetime(2017, 1, 1)  # Start from 2017 to get all available data
            since = int(start_dt.timestamp() * 1000)
        else:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            since = int(start_dt.timestamp() * 1000)

        if end_date is None:
            end_dt = datetime.now()
        else:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        if start_dt >= end_dt:
            raise DataDownloadError("start_date must be before end_date")

        # Estimate total candles based on date range and timeframe
        tf_minutes = TIMEFRAME_MINUTES.get(timeframe, 60)
        total_minutes = (end_dt - start_dt).total_seconds() / 60
        estimated_total = int(total_minutes / tf_minutes)

        all_candles = []
        current_since = since

        # Create progress bar
        with tqdm(total=estimated_total, desc=f"{symbol} {timeframe}", unit="candles") as pbar:
            while True:
                try:
                    # Fetch with retry logic
                    candles = self._fetch_with_retry(symbol, timeframe, current_since, limit)

                    if not candles:
                        break

                    all_candles.extend(candles)
                    
                    # Update progress bar
                    pbar.n = len(all_candles)
                    pbar.refresh()
                    
                    # Call progress callback if provided
                    if progress_callback:
                        progress_callback(len(all_candles))

                    # Move to next batch
                    last_candle_time = candles[-1][0]
                    last_candle_dt = datetime.fromtimestamp(last_candle_time / 1000)

                    # Stop if we've reached end date
                    if last_candle_dt >= end_dt:
                        break

                    # Update since for next iteration
                    current_since = last_candle_time + 1

                    # Rate limiting
                    time.sleep(self.rate_limit_ms / 1000.0)

                except Exception as e:
                    raise DataDownloadError(f"Failed to download {symbol} data: {str(e)}")
            
            # Update progress bar to final count
            pbar.n = len(all_candles)
            pbar.total = len(all_candles)
            pbar.refresh()

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
        Save downloaded futures data to CSV file.

        Args:
            data: DataFrame with OHLCV data
            symbol: Trading pair symbol (used in filename if not specified)
            timeframe: Candlestick timeframe (used in filename if not specified)
            filename: Custom filename (default: {symbol}_{timeframe}_{market_type}.csv)

        Returns:
            Path to saved CSV file

        Raises:
            DataDownloadError: If save fails
        """
        try:
            if filename is None:
                # Create filename from symbol, timeframe, and market type
                safe_symbol = symbol.replace("/", "_").replace(":", "_")
                filename = f"{safe_symbol}_{timeframe}_{self.market_type}.csv"

            filepath = self.data_dir / filename

            # Save to CSV
            data.to_csv(filepath, index=False)
            logger.info(f"Saved {self.market_type} data to {filepath}")

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
