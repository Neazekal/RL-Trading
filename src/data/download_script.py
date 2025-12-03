"""CLI script for downloading cryptocurrency market data."""

import logging
import sys
from typing import List
from datetime import datetime
from src.data.data_downloader import DataDownloader, DataDownloadError

logger = logging.getLogger(__name__)

# Default quote currency for converting coin names to trading pairs
DEFAULT_QUOTE_CURRENCY = "USDT"


def _normalize_symbol(symbol: str, quote_currency: str = DEFAULT_QUOTE_CURRENCY) -> str:
    """
    Normalize symbol to trading pair format.
    
    Converts coin names to trading pairs:
    - "BTC" -> "BTC/USDT"
    - "BTC/USDT" -> "BTC/USDT" (already in correct format)
    - "ETH" -> "ETH/USDT"
    
    Args:
        symbol: Coin name or trading pair
        quote_currency: Quote currency to use (default: USDT)
    
    Returns:
        Normalized trading pair symbol
    """
    # If already in trading pair format (contains /), return as-is
    if "/" in symbol:
        return symbol
    
    # Otherwise, convert coin name to trading pair
    return f"{symbol.upper()}/{quote_currency}"



def download_command(args) -> None:
    """
    Handle the download command from CLI.
    
    Args:
        args: Parsed command-line arguments containing:
            - symbols: List of trading symbols or coin names to download
            - timeframes: List of timeframes to download
            - exchange: Exchange name (default: binance)
            - market_type: Market type (spot or future)
            - start_date: Start date in YYYY-MM-DD format (default: earliest available)
            - end_date: End date in YYYY-MM-DD format (default: today)
            - data_dir: Directory to save data
    """
    try:
        # Initialize downloader
        logger.info(f"Initializing {args.exchange} downloader for {args.market_type} market...")
        downloader = DataDownloader(
            exchange_name=args.exchange,
            market_type=args.market_type,
            data_dir=args.data_dir,
        )
        
        # Normalize symbols (convert coin names to trading pairs)
        normalized_symbols = [_normalize_symbol(symbol) for symbol in args.symbols]
        logger.info(f"Normalized symbols: {', '.join(normalized_symbols)}")
        
        # Determine date range
        start_date = args.start_date  # None means earliest available data
        end_date = args.end_date if args.end_date else datetime.now().strftime("%Y-%m-%d")
        
        # If no start date specified, use earliest available (None will be handled by downloader)
        if start_date is None:
            logger.info("No start date specified. Will download from earliest available data for each coin.")
        
        successful = 0
        failed = 0
        
        logger.info(f"Starting download of {len(normalized_symbols)} coins with {len(args.timeframes)} timeframe(s)...")
        
        # Download each coin
        for symbol in normalized_symbols:
            for timeframe in args.timeframes:
                try:
                    # Download OHLCV data (tqdm progress bar is shown inside download_ohlcv)
                    df = downloader.download_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    
                    # Save to CSV
                    filepath = downloader.save_to_csv(
                        data=df,
                        symbol=symbol,
                        timeframe=timeframe,
                    )
                    
                    successful += 1
                    print(f"✓ Saved {len(df)} candles to {filepath}")
                    
                except DataDownloadError as e:
                    failed += 1
                    print(f"✗ Failed: {str(e)}")
                    continue
        
        # Summary
        total_downloads = len(normalized_symbols) * len(args.timeframes)
        logger.info(f"\n{'='*60}")
        logger.info(f"Download Summary:")
        logger.info(f"  Total: {total_downloads}")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Data saved to: {args.data_dir}/")
        logger.info(f"{'='*60}")
        
        if failed > 0:
            sys.exit(1)
        
    except DataDownloadError as e:
        logger.error(f"Download failed: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)
