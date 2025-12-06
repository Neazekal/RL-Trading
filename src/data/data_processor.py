"""Data processor for loading, transforming, and validating market data.

This module provides the DataProcessor class for:
- Loading OHLCV data from CSV files
- Calculating technical indicators (RSI, MACD, MA, Bollinger Bands, etc.)
- Transforming features using log-based engineering
- Validating data and handling missing values
- Serializing/deserializing MarketState objects
"""

import json
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.data_models import DataConfig, MarketState, ValidationResult


class DataProcessor:
    """Processes market data for the trading system.
    
    Handles data loading, indicator calculation, feature transformation,
    and validation. Separates model features (transformed) from raw prices
    (for P&L calculation).
    """
    
    # Feature column names after transformation
    FEATURE_COLUMNS = [
        # Price features
        'body', 'upper_wick', 'lower_wick', 'full_range', 'log_return',
        # Momentum
        'rsi_scaled', 'stoch_scaled',
        # Trend
        'ma20_rel', 'ma50_rel', 'ema12_rel', 'ma_ratio', 'macd_norm', 'hist_norm',
        # Volatility
        'bb_z', 'atr_rel',
        # Volume
        'vol_log', 'vma20_log', 'vol_rel_log', 'obv_sign', 'obv_mag_norm'
    ]
    
    def __init__(self, config: Optional[DataConfig] = None):
        """Initialize DataProcessor with optional configuration.
        
        Args:
            config: DataConfig with source, symbols, timeframe, indicators settings.
        """
        self.config = config
        # Z-score normalization parameters (calculated from training data)
        self._macd_mean: Optional[float] = None
        self._macd_std: Optional[float] = None
        self._hist_mean: Optional[float] = None
        self._hist_std: Optional[float] = None
        self._obv_mag_mean: Optional[float] = None
        self._obv_mag_std: Optional[float] = None
        self._is_fitted: bool = False
    
    def load_data(self, source: str, start_date: Optional[str] = None, 
                  end_date: Optional[str] = None) -> pd.DataFrame:
        """Load raw market data from CSV file.
        
        Args:
            source: Path to CSV file containing OHLCV data.
            start_date: Optional start date filter (YYYY-MM-DD format).
            end_date: Optional end date filter (YYYY-MM-DD format).
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume.
            
        Raises:
            FileNotFoundError: If source file doesn't exist.
            ValueError: If required columns are missing.
        """
        df = pd.read_csv(source)
        
        # Validate required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Apply date filters if provided
        if start_date:
            df = df[df['timestamp'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['timestamp'] <= pd.to_datetime(end_date)]
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators on OHLCV data.
        
        Calculates:
        - Momentum: RSI (14), Stochastic Oscillator (14)
        - Trend: MACD (12/26/9), SMA (20, 50), EMA (12)
        - Volatility: Bollinger Bands (20, 2σ), ATR (14)
        - Volume: VMA (20), OBV
        
        Args:
            data: DataFrame with OHLCV columns.
            
        Returns:
            DataFrame with original columns plus indicator columns.
        """
        df = data.copy()
        
        # ===== Momentum Indicators =====
        # RSI (14-period)
        df['rsi'] = self._calculate_rsi(df['close'], period=14)
        
        # Stochastic Oscillator (14-period)
        df['stoch'] = self._calculate_stochastic(df, period=14)
        
        # ===== Trend Indicators =====
        # Simple Moving Averages
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['sma50'] = df['close'].rolling(window=50).mean()
        
        # Exponential Moving Average (12-period)
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        
        # MACD (12/26/9)
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd_line'] = ema12 - ema26
        df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd_line'] - df['macd_signal']
        
        # ===== Volatility Indicators =====
        # Bollinger Bands (20-period, 2 std dev)
        df['bb_mid'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        
        # ATR (14-period)
        df['atr'] = self._calculate_atr(df, period=14)
        
        # ===== Volume Indicators =====
        # Volume Moving Average (20-period)
        df['vma20'] = df['volume'].rolling(window=20).mean()
        
        # On-Balance Volume (OBV)
        df['obv'] = self._calculate_obv(df)
        
        return df

    def transform_features(self, data: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Apply log-based feature engineering transformations.
        
        Transforms raw indicators into model-ready features using log-based
        transformations that don't require recalculation for new data.
        
        Args:
            data: DataFrame with OHLCV and indicator columns.
            fit: If True, calculate normalization parameters from this data.
                 Should be True for training data, False for validation/test.
            
        Returns:
            DataFrame with transformed feature columns.
        """
        df = data.copy()
        
        # Ensure indicators are calculated
        if 'rsi' not in df.columns:
            df = self.calculate_indicators(df)
        
        # ===== Price Features (log-based) =====
        log_open = np.log(df['open'].clip(lower=1e-10))
        log_high = np.log(df['high'].clip(lower=1e-10))
        log_low = np.log(df['low'].clip(lower=1e-10))
        log_close = np.log(df['close'].clip(lower=1e-10))
        
        df['body'] = log_close - log_open
        df['upper_wick'] = log_high - np.maximum(log_open, log_close)
        df['lower_wick'] = np.minimum(log_open, log_close) - log_low
        df['full_range'] = log_high - log_low
        df['log_return'] = log_close.diff()
        
        # ===== Momentum Features =====
        df['rsi_scaled'] = (df['rsi'] - 50) / 50
        df['stoch_scaled'] = (df['stoch'] - 50) / 50
        
        # ===== Trend Features =====
        # Relative position to moving averages (log-based)
        df['ma20_rel'] = np.log((df['close'] / df['sma20']).clip(lower=1e-10))
        df['ma50_rel'] = np.log((df['close'] / df['sma50']).clip(lower=1e-10))
        df['ema12_rel'] = np.log((df['close'] / df['ema12']).clip(lower=1e-10))
        df['ma_ratio'] = np.log((df['sma20'] / df['sma50']).clip(lower=1e-10))
        
        # MACD normalization (Z-score with training parameters)
        if fit:
            self._macd_mean = df['macd_line'].mean()
            self._macd_std = df['macd_line'].std()
            self._hist_mean = df['macd_hist'].mean()
            self._hist_std = df['macd_hist'].std()
            if self._macd_std == 0:
                self._macd_std = 1.0
            if self._hist_std == 0:
                self._hist_std = 1.0
            self._is_fitted = True
        
        if not self._is_fitted:
            raise ValueError("DataProcessor must be fitted on training data first. "
                           "Call transform_features with fit=True on training data.")
        
        df['macd_norm'] = (df['macd_line'] - self._macd_mean) / self._macd_std
        df['hist_norm'] = (df['macd_hist'] - self._hist_mean) / self._hist_std
        
        # ===== Volatility Features =====
        # Bollinger Band Z-score
        bb_std_safe = df['bb_std'].replace(0, 1)
        df['bb_z'] = (df['close'] - df['bb_mid']) / (2 * bb_std_safe)
        
        # ATR relative to price (log-based)
        df['atr_rel'] = np.log(df['atr'].clip(lower=1e-10)) - log_close
        
        # ===== Volume Features =====
        df['vol_log'] = np.log(df['volume'] + 1)
        df['vma20_log'] = np.log(df['vma20'] + 1)
        df['vol_rel_log'] = df['vol_log'] - df['vma20_log']
        
        # OBV features
        df['obv_sign'] = np.sign(df['obv'])
        obv_mag = np.log(np.abs(df['obv']) + 1)
        
        if fit:
            self._obv_mag_mean = obv_mag.mean()
            self._obv_mag_std = obv_mag.std()
            if self._obv_mag_std == 0:
                self._obv_mag_std = 1.0
        
        df['obv_mag_norm'] = (obv_mag - self._obv_mag_mean) / self._obv_mag_std
        
        return df
    
    def get_model_features(self, data: pd.DataFrame) -> np.ndarray:
        """Get transformed features for neural network input.
        
        Args:
            data: DataFrame with transformed feature columns.
            
        Returns:
            NumPy array of shape (n_samples, n_features) with transformed features.
        """
        # Ensure features are transformed
        if 'body' not in data.columns:
            raise ValueError("Data must be transformed first. Call transform_features().")
        
        # Select feature columns
        features = data[self.FEATURE_COLUMNS].values
        
        return features
    
    def get_raw_prices(self, data: pd.DataFrame) -> pd.DataFrame:
        """Get raw close prices for environment P&L calculation.
        
        Args:
            data: DataFrame with OHLCV data.
            
        Returns:
            DataFrame with timestamp and close price columns.
        """
        return data[['timestamp', 'close']].copy()
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, ValidationResult]:
        """Validate data completeness and handle missing values.
        
        Applies forward-fill interpolation for missing values.
        
        Args:
            data: DataFrame to validate.
            
        Returns:
            Tuple of (cleaned DataFrame, ValidationResult).
        """
        df = data.copy()
        errors = []
        warnings = []
        
        # Check for required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return df, ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        
        # Count missing values before fill
        total_cells = len(df) * len(df.columns)
        missing_before = df.isnull().sum().sum()
        missing_pct = (missing_before / total_cells) * 100 if total_cells > 0 else 0
        
        if missing_pct > 5:
            warnings.append(f"High missing data rate: {missing_pct:.2f}%")
        
        # Forward-fill missing values
        df = df.ffill()
        
        # Back-fill any remaining NaN at the start
        df = df.bfill()
        
        # Check for any remaining NaN
        remaining_nan = df.isnull().sum().sum()
        if remaining_nan > 0:
            errors.append(f"Unable to fill {remaining_nan} missing values")
            return df, ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        
        # Validate price relationships (high >= low, etc.)
        invalid_prices = (df['high'] < df['low']).sum()
        if invalid_prices > 0:
            warnings.append(f"Found {invalid_prices} rows where high < low")
        
        # Validate non-negative volume
        negative_volume = (df['volume'] < 0).sum()
        if negative_volume > 0:
            warnings.append(f"Found {negative_volume} rows with negative volume")
            df.loc[df['volume'] < 0, 'volume'] = 0
        
        is_valid = len(errors) == 0
        return df, ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)

    def to_json(self, state: MarketState) -> str:
        """Serialize MarketState to JSON string.
        
        Args:
            state: MarketState object to serialize.
            
        Returns:
            JSON string representation.
        """
        return json.dumps(state.to_dict())
    
    def from_json(self, json_str: str) -> MarketState:
        """Deserialize JSON string to MarketState.
        
        Args:
            json_str: JSON string to deserialize.
            
        Returns:
            MarketState object.
        """
        data = json.loads(json_str)
        return MarketState.from_dict(data)
    
    # ===== Private Helper Methods =====
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index.
        
        Args:
            prices: Series of closing prices.
            period: RSI period (default 14).
            
        Returns:
            Series of RSI values (0-100).
        """
        delta = prices.diff()
        
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_stochastic(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Stochastic Oscillator %K.
        
        Args:
            data: DataFrame with high, low, close columns.
            period: Stochastic period (default 14).
            
        Returns:
            Series of Stochastic %K values (0-100).
        """
        lowest_low = data['low'].rolling(window=period).min()
        highest_high = data['high'].rolling(window=period).max()
        
        denominator = (highest_high - lowest_low).replace(0, 1e-10)
        stoch_k = 100 * (data['close'] - lowest_low) / denominator
        
        return stoch_k
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range.
        
        Args:
            data: DataFrame with high, low, close columns.
            period: ATR period (default 14).
            
        Returns:
            Series of ATR values.
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        return atr
    
    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume.
        
        Args:
            data: DataFrame with close and volume columns.
            
        Returns:
            Series of OBV values.
        """
        close = data['close']
        volume = data['volume']
        
        # Direction: +1 if close > prev close, -1 if close < prev close, 0 if equal
        direction = np.sign(close.diff())
        
        # OBV is cumulative sum of signed volume
        obv = (direction * volume).cumsum()
        
        return obv
    
    def fit(self, data: pd.DataFrame) -> 'DataProcessor':
        """Fit the processor on training data.
        
        Calculates normalization parameters from training data.
        
        Args:
            data: Training DataFrame with OHLCV data.
            
        Returns:
            Self for method chaining.
        """
        # Calculate indicators
        df = self.calculate_indicators(data)
        
        # Transform features with fit=True to calculate normalization params
        self.transform_features(df, fit=True)
        
        return self
    
    def get_normalization_params(self) -> dict:
        """Get normalization parameters for persistence.
        
        Returns:
            Dictionary of normalization parameters.
        """
        if not self._is_fitted:
            raise ValueError("DataProcessor has not been fitted yet.")
        
        return {
            'macd_mean': self._macd_mean,
            'macd_std': self._macd_std,
            'hist_mean': self._hist_mean,
            'hist_std': self._hist_std,
            'obv_mag_mean': self._obv_mag_mean,
            'obv_mag_std': self._obv_mag_std,
        }
    
    def set_normalization_params(self, params: dict) -> None:
        """Set normalization parameters from persistence.
        
        Args:
            params: Dictionary of normalization parameters.
        """
        self._macd_mean = params['macd_mean']
        self._macd_std = params['macd_std']
        self._hist_mean = params['hist_mean']
        self._hist_std = params['hist_std']
        self._obv_mag_mean = params['obv_mag_mean']
        self._obv_mag_std = params['obv_mag_std']
        self._is_fitted = True
    
    def save_processed_data(
        self,
        data: pd.DataFrame,
        output_path: str,
        features_only: bool = False,
        include_close: bool = True,
    ) -> str:
        """Save processed data to CSV file.
        
        Args:
            data: DataFrame with processed data (indicators and/or features).
            output_path: Path to save the CSV file.
            features_only: If True, save only the feature columns used by the model
                          (plus timestamp, close for P&L, and atr for SL/TP).
            include_close: If True and features_only=True, include close price
                          for environment P&L calculation.
        
        Returns:
            Path to the saved file.
        """
        if features_only:
            # Save timestamp, features, close for P&L, and atr for SL/TP calculation
            cols_to_save = ['timestamp'] + self.FEATURE_COLUMNS
            if include_close:
                cols_to_save.append('close')
            # Always include raw ATR for environment SL/TP calculation
            cols_to_save.append('atr')
            available_cols = [c for c in cols_to_save if c in data.columns]
            data[available_cols].to_csv(output_path, index=False)
        else:
            data.to_csv(output_path, index=False)
        
        return output_path
    
    def process_and_save(
        self,
        source: str,
        output_path: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        features_only: bool = False,
        drop_null: bool = True,
        include_close: bool = True,
    ) -> pd.DataFrame:
        """Load, process, and save data in one step.
        
        Args:
            source: Path to input CSV file.
            output_path: Path to save processed CSV file.
            start_date: Optional start date filter.
            end_date: Optional end date filter.
            features_only: If True, save only feature columns (+ close for P&L).
            drop_null: If True, drop rows with null values (from indicator warm-up).
            include_close: If True and features_only=True, include close for P&L.
        
        Returns:
            Processed DataFrame.
        """
        # Load data
        df = self.load_data(source, start_date, end_date)
        
        # Validate and clean
        df, _ = self.validate_data(df)
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Transform features
        df = self.transform_features(df, fit=True)
        
        # Drop null rows if requested (removes warm-up period)
        if drop_null:
            df = df.dropna().reset_index(drop=True)
        
        # Save to file
        self.save_processed_data(df, output_path, features_only, include_close)
        
        return df
