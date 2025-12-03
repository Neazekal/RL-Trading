# RL-Trading

A dual-agent reinforcement learning system for cryptocurrency trading.

## Quick Start

### Download Market Data

Download cryptocurrency market data from exchanges like Binance or Bybit:

```bash
# Download BTC and ETH data for 1h timeframe (from earliest available to today)
# Note: "Earliest available" means from 2017-01-01 or the coin's listing date, whichever is later
python main.py download --symbols BTC ETH --timeframes 1h

# Download multiple timeframes
python main.py download --symbols BTC --timeframes 1m 5m 15m 1h 4h 1d

# Download with specific date range
python main.py download --symbols BTC --timeframes 1h --start-date 2023-01-01 --end-date 2023-12-31

# Download from Bybit exchange (spot market)
python main.py download --symbols BTC --exchange bybit --market-type spot

# Custom data directory
python main.py download --symbols BTC ETH ADA --data-dir ./market_data

# You can also use full trading pair format if needed
python main.py download --symbols BTC/USDT ETH/USDT --timeframes 1h
```

The download command will:
- Download data for all symbol-timeframe combinations
- Show download progress for each coin (number of candles downloaded)
- Save data to organized CSV files in the `data/` directory
- Display a summary of successful and failed downloads with detailed logs

Example output:
```
BTC/USDT 1h: 100%|████████████████████████████████| 8065/8065 [00:12<00:00, 672.08candles/s]
✓ Saved 8065 candles to data/BTC_USDT_1h_future.csv

ETH/USDT 1h: 100%|████████████████████████████████| 8065/8065 [00:11<00:00, 733.18candles/s]
✓ Saved 8065 candles to data/ETH_USDT_1h_future.csv

============================================================
Download Summary:
  Total: 2
  Successful: 2
  Failed: 0
  Data saved to: data/
============================================================
```

### Features

- **Dual-Agent Architecture**: Separate agents for opening and closing positions
- **Signal Confirmation**: N-signal confirmation layer to reduce noise
- **PPO Algorithm**: Proximal Policy Optimization for stable training
- **Backtesting**: Historical simulation with realistic fees and slippage
- **Flexible Configuration**: YAML-based configuration management