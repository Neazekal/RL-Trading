"""CLI script for processing market data.

Usage:
    python -m src.data.process_script BTC --timeframe 1h
    python -m src.data.process_script BTC DOGE TRX --timeframe 1h
    python -m src.data.process_script BTC --timeframe 1h --features-only
"""

import argparse
import sys
from pathlib import Path

from src.data.data_processor import DataProcessor


def main():
    parser = argparse.ArgumentParser(description="Process market data with indicators and features")
    parser.add_argument("symbols", nargs="+", help="Coin symbols (e.g., BTC DOGE TRX)")
    parser.add_argument("--timeframe", "-t", default="1h", help="Timeframe (default: 1h)")
    parser.add_argument("--market", "-m", default="future", help="Market type: spot or future (default: future)")
    parser.add_argument("--features-only", "-f", action="store_true", help="Save only feature columns")
    parser.add_argument("--start-date", "-s", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", "-e", help="End date (YYYY-MM-DD)")
    parser.add_argument("--keep-null", action="store_true", help="Keep rows with null values (warm-up period)")
    
    args = parser.parse_args()
    
    dp = DataProcessor()
    
    for symbol in args.symbols:
        symbol = symbol.upper()
        input_file = f"data/{symbol}_USDT_{args.timeframe}_{args.market}.csv"
        output_file = f"data/{symbol}_USDT_{args.timeframe}_processed.csv"
        
        if not Path(input_file).exists():
            print(f"Error: {input_file} not found")
            continue
        
        print(f"Processing {input_file}...")
        df = dp.process_and_save(
            input_file, 
            output_file,
            start_date=args.start_date,
            end_date=args.end_date,
            features_only=args.features_only,
            drop_null=not args.keep_null
        )
        print(f"Saved {len(df)} rows to {output_file}")


if __name__ == "__main__":
    main()
