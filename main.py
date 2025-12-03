"""
RL Crypto Trading Agents - Main Entry Point

A dual-agent reinforcement learning system for cryptocurrency trading.
"""

import argparse
import sys
import logging
from src.data.download_script import download_command


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    """Main entry point for the RL trading system."""
    parser = argparse.ArgumentParser(
        description="RL Crypto Trading Agents - Train and backtest RL trading agents"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download cryptocurrency market data")
    download_parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=["BTC", "ETH"],
        help="Coin names or trading symbols to download (e.g., BTC ETH or BTC/USDT ETH/USDT)"
    )
    download_parser.add_argument(
        "--timeframes",
        type=str,
        nargs="+",
        default=["1h"],
        help="Timeframes to download (e.g., 1m 5m 15m 1h 4h 1d)"
    )
    download_parser.add_argument(
        "--exchange",
        type=str,
        default="binance",
        help="Exchange to download from (default: binance)"
    )
    download_parser.add_argument(
        "--market-type",
        type=str,
        default="future",
        choices=["spot", "future"],
        help="Market type to download (default: future)"
    )
    download_parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date in format YYYY-MM-DD (default: earliest available data)"
    )
    download_parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date in format YYYY-MM-DD (default: today)"
    )
    download_parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to save downloaded data (default: data)"
    )
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the trading agents")
    train_parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration file"
    )
    train_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    
    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtesting")
    backtest_parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration file"
    )
    backtest_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    
    # Inference command
    inference_parser = subparsers.add_parser("inference", help="Run inference mode")
    inference_parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration file"
    )
    inference_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate command handler
    if args.command == "download":
        download_command(args)
    else:
        # TODO: Implement command handlers in future tasks
        print(f"Command '{args.command}' will be implemented in future tasks.")
        if hasattr(args, 'config'):
            print(f"Using config: {args.config}")


if __name__ == "__main__":
    main()
