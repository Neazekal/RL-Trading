"""Core data models and dataclasses for the trading system."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Literal, Optional, Tuple
import numpy as np


# ============================================================================
# Core Trading Data Models
# ============================================================================

@dataclass
class MarketState:
    """Represents a single market observation (OHLCV + indicators)."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    rsi: float
    macd: float
    macd_signal: float
    ma_short: float
    ma_long: float
    normalized_close: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'open': float(self.open),
            'high': float(self.high),
            'low': float(self.low),
            'close': float(self.close),
            'volume': float(self.volume),
            'rsi': float(self.rsi),
            'macd': float(self.macd),
            'macd_signal': float(self.macd_signal),
            'ma_short': float(self.ma_short),
            'ma_long': float(self.ma_long),
            'normalized_close': float(self.normalized_close),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'MarketState':
        """Create from dictionary (JSON deserialization)."""
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            volume=data['volume'],
            rsi=data['rsi'],
            macd=data['macd'],
            macd_signal=data['macd_signal'],
            ma_short=data['ma_short'],
            ma_long=data['ma_long'],
            normalized_close=data['normalized_close'],
        )


@dataclass
class Position:
    """Represents an open trading position."""
    id: str
    direction: Literal["long", "short"]
    entry_price: float
    entry_time: datetime
    size: float


@dataclass
class TradeResult:
    """Represents a completed trade with P&L calculation."""
    position_id: str
    entry_price: float
    exit_price: float
    direction: str
    size: float
    profit_loss: float
    profit_loss_percent: float
    fees: float
    duration: timedelta


@dataclass
class Experience:
    """Represents a single experience tuple for training."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float


@dataclass
class EpisodeResult:
    """Results from a single training episode."""
    total_reward: float
    num_trades: int
    win_rate: float
    final_balance: float
    experiences: List[Experience]


@dataclass
class PerformanceMetrics:
    """Performance metrics for backtesting and evaluation."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_profit_per_trade: float
    num_trades: int
    profit_factor: float


# ============================================================================
# Configuration Dataclasses
# ============================================================================

@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    source: str
    symbols: List[str]
    timeframe: str
    indicators: List[str]


@dataclass
class EnvConfig:
    """Configuration for the trading environment."""
    initial_balance: float
    max_positions: int = 1  # Single position constraint
    trading_fee: float = 0.001
    slippage: float = 0.0005


@dataclass
class AgentConfig:
    """Configuration for agent training."""
    learning_rate: float
    gamma: float  # discount factor
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: float
    clip_ratio: float  # PPO clip parameter


@dataclass
class ActorNetworkConfig:
    """Configuration for Actor network architecture."""
    input_size: int
    hidden_sizes: List[int]
    lstm_hidden_size: int
    lstm_num_layers: int
    num_actions: int


@dataclass
class CriticNetworkConfig:
    """Configuration for Critic network architecture."""
    input_size: int
    hidden_sizes: List[int]
    lstm_hidden_size: int
    lstm_num_layers: int
    # Output is always 1 (scalar value estimate)


@dataclass
class ActorCriticConfig:
    """Configuration for Actor-Critic module."""
    actor: ActorNetworkConfig
    critic: CriticNetworkConfig
    gae_lambda: float = 0.95  # GAE lambda for advantage estimation


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    num_episodes: int
    batch_size: int
    buffer_capacity: int
    train_val_ratio: float
    early_stopping_patience: int
    checkpoint_frequency: int


@dataclass
class RewardConfig:
    """Configuration for reward function."""
    open_penalty: float
    profit_multiplier: float
    loss_multiplier: float
    discount_factor: float


@dataclass
class SignalConfirmationConfig:
    """Configuration for signal confirmation layer."""
    required_confirmations: int  # N consecutive signals required
    enabled: bool = True  # Whether confirmation layer is active


# ============================================================================
# Additional Data Structures
# ============================================================================

@dataclass
class CloseDecision:
    """Decision from closing agent about a position."""
    position_id: str
    should_close: bool
    confidence: float


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    trades: List[TradeResult]
    metrics: PerformanceMetrics
    final_balance: float
    initial_balance: float


@dataclass
class BacktestReport:
    """Detailed backtest report with trade logs."""
    result: BacktestResult
    trade_logs: List[dict]
    summary: dict


@dataclass
class ComparisonReport:
    """Report comparing multiple strategy results."""
    results: List[BacktestResult]
    best_strategy: int
    comparison_metrics: dict


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
