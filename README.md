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

### Process Market Data

Transform raw OHLCV data into model-ready features with technical indicators:

```bash
# Process BTC data (drops null rows from warm-up period by default)
python -m src.data.process_script BTC

# Process multiple coins
python -m src.data.process_script BTC DOGE TRX

# Save only features (excludes raw OHLC, keeps close for P&L)
python -m src.data.process_script BTC --features-only

# Keep null rows from indicator warm-up period
python -m src.data.process_script BTC --keep-null

# With date range
python -m src.data.process_script BTC --start-date 2024-01-01 --end-date 2024-12-31

# Custom timeframe
python -m src.data.process_script BTC --timeframe 4h
```

The DataProcessor will:
- Calculate technical indicators (RSI, MACD, SMA, EMA, Bollinger Bands, ATR, OBV, etc.)
- Transform features using log-based engineering (20 features total)
- Validate data and handle missing values with forward-fill
- Save processed data to CSV with indicators and features
- Drop warm-up period rows (first ~50 rows with NaN from indicator calculation)

Example output:
```
Processing data/BTC_USDT_1h_future.csv...
Saved 54607 rows to data/BTC_USDT_1h_processed.csv
```

### Features

- **Dual-Agent Architecture**: Separate agents for opening and closing positions
- **Signal Confirmation**: N-signal confirmation layer to reduce noise
- **PPO Algorithm**: Proximal Policy Optimization for stable training
- **Data Processing**: 
  - Technical indicators: RSI, Stochastic, MACD, SMA, EMA, Bollinger Bands, ATR, OBV
  - Log-based feature engineering with 20 transformed features
  - Data validation and missing value handling
  - JSON serialization for MarketState
- **Backtesting**: Historical simulation with realistic fees and slippage
- **Flexible Configuration**: YAML-based configuration management


## Implementation Status

### ✅ Task 4: Data Processor (Completed)

#### 4.1 DataProcessor Class
The `DataProcessor` class (`src/data/data_processor.py`) provides comprehensive market data processing:

**Methods:**
- `load_data(source, start_date, end_date)` - Load OHLCV data from CSV
- `calculate_indicators(data)` - Calculate all technical indicators
- `transform_features(data, fit)` - Apply log-based feature engineering
- `get_model_features(data)` - Get 20 transformed features for neural network
- `get_raw_prices(data)` - Get raw close prices for P&L calculation
- `validate_data(data)` - Validate and forward-fill missing values
- `to_json(state)` / `from_json(json_str)` - Serialize/deserialize MarketState
- `save_processed_data(data, output_path, features_only)` - Save to CSV
- `process_and_save(source, output_path, ...)` - One-step processing pipeline
- `fit(data)` - Fit normalization parameters on training data
- `get_normalization_params()` / `set_normalization_params(params)` - Persistence

**Technical Indicators Calculated:**
- **Momentum**: RSI (14-period), Stochastic Oscillator (14-period)
- **Trend**: MACD (12/26/9), SMA (20, 50), EMA (12)
- **Volatility**: Bollinger Bands (20-period, 2σ), ATR (14-period)
- **Volume**: Volume Moving Average (20-period), On-Balance Volume (OBV)

**Feature Engineering (20 Features):**
- **Price Features** (5): body, upper_wick, lower_wick, full_range, log_return
- **Momentum** (2): rsi_scaled, stoch_scaled
- **Trend** (6): ma20_rel, ma50_rel, ema12_rel, ma_ratio, macd_norm, hist_norm
- **Volatility** (2): bb_z, atr_rel
- **Volume** (5): vol_log, vma20_log, vol_rel_log, obv_sign, obv_mag_norm

All features use log-based transformations with Z-score normalization for MACD and OBV magnitude.

#### 4.2 Property Test: Feature Transformation Validity
Tests that log-based transformations produce finite values for valid OHLCV data:
- All transformed features are finite (no NaN/Inf)
- Raw prices preserved for P&L calculation
- Feature count matches specification (20 features)
- Scaled features in expected ranges

**Status**: ✅ Passed (4 tests, 100 examples each)

#### 4.3 Property Test: Missing Data Forward-Fill
Tests that missing values are properly handled:
- No missing values after validation
- Filled values equal most recent non-missing value
- Non-missing values preserved
- Row count preserved

**Status**: ✅ Passed (4 tests, 100 examples each)

#### CLI Script: process_script.py
Easy-to-use command-line interface for data processing:

```bash
# Basic usage
python -m src.data.process_script BTC

# Multiple coins
python -m src.data.process_script BTC DOGE TRX

# Options
--timeframe 1h          # Timeframe (default: 1h)
--market future         # Market type: spot or future (default: future)
--features-only         # Save only feature columns
--keep-null             # Keep rows with null values (warm-up period)
--start-date 2024-01-01 # Start date filter
--end-date 2024-12-31   # End date filter
```

#### Test Results
All 53 tests passing:
- 4 property tests for feature transformation validity
- 4 property tests for missing data forward-fill
- 4 property tests for MarketState serialization
- 41 unit tests for data downloader and download script

#### Files Added/Modified
- ✅ `src/data/data_processor.py` - DataProcessor class (564 lines)
- ✅ `src/data/process_script.py` - CLI script (52 lines)
- ✅ `tests/property/test_feature_transformation.py` - Property tests
- ✅ `tests/property/test_missing_data_forward_fill.py` - Property tests
- ✅ `src/data/__init__.py` - Export DataProcessor
- ✅ `src/models/data_models.py` - Fixed duplicate field

### Next Tasks

- [ ] Task 5: Position Manager
- [ ] Task 6: Signal Confirmation Layer
- [ ] Task 7: Trading Environment
- [ ] Task 8: Reward Function
- [ ] Task 9: Actor-Critic Networks
- [ ] Task 10: Order Opening Agent
- [ ] Task 11: Order Closing Agent
- [ ] Task 12: Replay Buffer
- [ ] Task 13: Model Persistence
- [ ] Task 14: Configuration Manager
- [ ] Task 15: Training Pipeline
- [ ] Task 16: Backtesting System
- [ ] Task 17: Main Entry Point
