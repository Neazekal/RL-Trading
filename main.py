"""
RL Crypto Trading Agents - Main Entry Point

A dual-agent reinforcement learning system for cryptocurrency trading.
"""

import argparse
import sys


def main():
    """Main entry point for the RL trading system."""
    parser = argparse.ArgumentParser(
        description="RL Crypto Trading Agents - Train and backtest RL trading agents"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
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
    
    # TODO: Implement command handlers in future tasks
    print(f"Command '{args.command}' will be implemented in future tasks.")
    print(f"Using config: {args.config}")


if __name__ == "__main__":
    main()
