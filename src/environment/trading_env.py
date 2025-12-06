"""Gymnasium-compatible Trading Environment for RL agents.

This environment supports dual-agent trading with separate action spaces
for opening and closing positions. It enforces a single position constraint.
"""

from typing import Any, Dict, List, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from src.environment.position_manager import PositionManager
from src.models.data_models import EnvConfig, Position, TradeResult


class TradingEnvironment(gym.Env):
    """
    Gymnasium-compatible trading environment for dual-agent RL.
    
    The environment coordinates between two agents:
    - Order Opening Agent: 3 actions (0=hold, 1=open_long, 2=open_short)
    - Order Closing Agent: 2 actions (0=hold, 1=close)
    
    Single position constraint: only one position allowed at a time.
    
    Attributes:
        opening_action_space: Discrete(3) for opening agent
        closing_action_space: Discrete(2) for closing agent
        observation_space: Box space for market state features
    """
    
    # Action constants for Opening Agent
    OPEN_HOLD = 0
    OPEN_LONG = 1
    OPEN_SHORT = 2
    
    # Action constants for Closing Agent
    CLOSE_HOLD = 0
    CLOSE_POSITION = 1
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        config: EnvConfig,
        data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ):
        """
        Initialize the trading environment.
        
        Args:
            config: Environment configuration
            data: Market data DataFrame with OHLCV and features
            feature_columns: List of column names to use as observation features.
                           If None, uses all numeric columns except OHLCV.
        """
        super().__init__()
        
        self.config = config
        self.data = data.reset_index(drop=True)
        
        # Determine feature columns for observation
        if feature_columns is not None:
            self.feature_columns = feature_columns
        else:
            # Default: use all numeric columns
            self.feature_columns = list(data.select_dtypes(include=[np.number]).columns)
        
        # Validate data
        if len(self.data) == 0:
            raise ValueError("Data cannot be empty")
        
        # Action spaces for dual agents
        self.opening_action_space = spaces.Discrete(3)  # hold, open_long, open_short
        self.closing_action_space = spaces.Discrete(2)  # hold, close
        
        # Default action space (for Gymnasium compatibility)
        self.action_space = self.opening_action_space
        
        # Observation space: market features
        num_features = len(self.feature_columns)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_features,),
            dtype=np.float32
        )
        
        # Capital and risk management from config
        self.initial_capital = config.initial_capital
        self.risk_per_trade = config.risk_per_trade
        self.leverage = config.leverage
        self.trading_fee = config.trading_fee
        self.slippage = config.slippage
        
        # ATR-based SL/TP configuration (fixed at entry time)
        # Default: 1.5x ATR for SL, 3x ATR for TP (1:2 ratio, max 1:3)
        self.use_atr_sl_tp = True
        self.atr_sl_multiplier = 1.5
        self.atr_tp_multiplier = 3.0
        
        # Position manager
        self.position_manager = PositionManager(trading_fee=self.trading_fee)
        
        # State variables (initialized in reset)
        self.current_capital: float = self.initial_capital
        self.current_step: int = 0
        self.current_position: Optional[Position] = None
        self.episode_trades: List[TradeResult] = []
        self.cumulative_reward: float = 0.0
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
            
        Returns:
            Tuple of (observation, info dict)
        """
        super().reset(seed=seed)
        
        # Reset capital to initial value
        self.current_capital = self.initial_capital
        
        # Reset step counter
        self.current_step = 0
        
        # Clear position manager and current position
        self.position_manager.clear_all_positions()
        self.current_position = None
        
        # Clear episode tracking
        self.episode_trades = []
        self.cumulative_reward = 0.0
        
        # Get initial observation
        observation = self._get_observation()
        
        info = {
            "balance": self.current_capital,
            "position": None,
            "step": self.current_step
        }
        
        return observation, info

    def step(
        self,
        action: int,
        agent_type: str = "opening"
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute an action and return the new state.
        
        Args:
            action: Action to execute
            agent_type: "opening" or "closing" to specify which agent is acting
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        reward = 0.0
        trade_result = None
        action_taken = "hold"
        
        # Get current price for position operations
        current_price = self._get_current_price()
        
        if agent_type == "opening":
            reward, action_taken = self._process_opening_action(action, current_price)
        elif agent_type == "closing":
            reward, action_taken, trade_result = self._process_closing_action(action, current_price)
        else:
            raise ValueError(f"Invalid agent_type: {agent_type}. Must be 'opening' or 'closing'")
        
        # Advance to next step
        self.current_step += 1
        
        # Check if episode is done
        terminated = self._is_terminated()
        truncated = False
        
        # Update cumulative reward
        self.cumulative_reward += reward
        
        # Get new observation
        observation = self._get_observation()
        
        info = {
            "balance": self.current_capital,
            "position": self._position_to_dict(self.current_position),
            "step": self.current_step,
            "action_taken": action_taken,
            "trade_result": trade_result,
            "cumulative_reward": self.cumulative_reward,
            "current_price": current_price,
            "atr": self.get_current_atr()  # For dynamic SL/TP calculation
        }
        
        return observation, reward, terminated, truncated, info
    
    def _process_opening_action(
        self,
        action: int,
        current_price: float
    ) -> Tuple[float, str]:
        """
        Process an action from the opening agent.
        
        Args:
            action: Opening action (0=hold, 1=open_long, 2=open_short)
            current_price: Current market price
            
        Returns:
            Tuple of (reward, action_taken_string)
        """
        reward = 0.0
        action_taken = "hold"
        
        if action == self.OPEN_HOLD:
            action_taken = "hold"
            reward = 0.0
            
        elif action in (self.OPEN_LONG, self.OPEN_SHORT):
            # Check single position constraint
            if self.has_open_position():
                # Reject open action, treat as hold (Property 8)
                action_taken = "hold_rejected"
                reward = 0.0
            elif self.can_open_position():
                # Open new position
                direction = "long" if action == self.OPEN_LONG else "short"
                position_size = self._calculate_position_size(current_price)
                
                if position_size > 0:
                    # Apply slippage to entry price
                    slippage_multiplier = 1 + self.slippage if direction == "long" else 1 - self.slippage
                    entry_price = current_price * slippage_multiplier
                    
                    # Calculate fixed SL/TP prices at entry using ATR
                    sl_price, tp_price = self._calculate_entry_sl_tp(entry_price, direction)
                    
                    # Open position with fixed SL/TP
                    self.current_position = self.position_manager.open_position(
                        direction=direction,
                        price=entry_price,
                        size=position_size,
                        stop_loss_price=sl_price,
                        take_profit_price=tp_price
                    )
                    
                    action_taken = f"open_{direction}"
                    # Small negative reward for opening (to discourage excessive trading)
                    reward = -0.001  # Configurable via RewardConfig later
                else:
                    action_taken = "hold_insufficient_balance"
                    reward = 0.0
        
        return reward, action_taken
    
    def _process_closing_action(
        self,
        action: int,
        current_price: float
    ) -> Tuple[float, str, Optional[TradeResult]]:
        """
        Process an action from the closing agent.
        
        Args:
            action: Closing action (0=hold, 1=close)
            current_price: Current market price
            
        Returns:
            Tuple of (reward, action_taken_string, trade_result or None)
        """
        reward = 0.0
        action_taken = "hold"
        trade_result = None
        
        if action == self.CLOSE_HOLD:
            action_taken = "hold"
            reward = 0.0
            
        elif action == self.CLOSE_POSITION:
            if self.has_open_position() and self.current_position is not None:
                # Apply slippage to exit price
                direction = self.current_position.direction
                slippage_multiplier = 1 - self.slippage if direction == "long" else 1 + self.slippage
                exit_price = current_price * slippage_multiplier
                
                # Close position
                trade_result = self.position_manager.close_position(
                    position_id=self.current_position.id,
                    exit_price=exit_price
                )
                
                # Update capital
                self.current_capital += trade_result.profit_loss
                
                # Record trade
                self.episode_trades.append(trade_result)
                
                # Clear current position
                self.current_position = None
                
                action_taken = "close"
                # Reward proportional to P&L
                reward = trade_result.profit_loss_percent / 100.0  # Normalize
            else:
                # No position to close, treat as hold
                action_taken = "hold_no_position"
                reward = 0.0
        
        return reward, action_taken, trade_result
    
    def _calculate_position_size(self, current_price: float) -> float:
        """
        Calculate position size based on risk per trade and leverage.
        
        Args:
            current_price: Current market price
            
        Returns:
            Position size (quantity)
        """
        # Risk amount in capital
        risk_amount = self.current_capital * self.risk_per_trade
        
        # Position value with leverage
        position_value = risk_amount * self.leverage
        
        # Convert to quantity
        if current_price > 0:
            position_size = position_value / current_price
        else:
            position_size = 0.0
        
        return position_size
    
    def _calculate_entry_sl_tp(
        self,
        entry_price: float,
        direction: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate fixed SL/TP prices at entry time using ATR.
        
        SL/TP are calculated once at entry and stored in the Position.
        They do not change during the life of the position.
        
        Args:
            entry_price: Position entry price
            direction: "long" or "short"
            
        Returns:
            Tuple of (stop_loss_price, take_profit_price) or (None, None) if ATR unavailable
        """
        if not self.use_atr_sl_tp:
            return None, None
        
        atr = self.get_current_atr()
        if atr is None or atr <= 0:
            return None, None
        
        sl_distance = atr * self.atr_sl_multiplier
        tp_distance = atr * self.atr_tp_multiplier
        
        if direction == "long":
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        else:  # short
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance
        
        return sl_price, tp_price
    
    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation (market state features).
        
        Returns:
            Numpy array of features
        """
        if self.current_step >= len(self.data):
            # Return last observation if past data end
            step = len(self.data) - 1
        else:
            step = self.current_step
        
        row = self.data.iloc[step]
        features = row[self.feature_columns].values.astype(np.float32)
        
        return features
    
    def _get_current_price(self) -> float:
        """
        Get the current close price.
        
        Returns:
            Current close price
        """
        if self.current_step >= len(self.data):
            step = len(self.data) - 1
        else:
            step = self.current_step
        
        # Try to get 'close' column, fall back to 'Close'
        if 'close' in self.data.columns:
            return float(self.data.iloc[step]['close'])
        elif 'Close' in self.data.columns:
            return float(self.data.iloc[step]['Close'])
        else:
            # Use first numeric column as fallback
            return float(self.data.iloc[step][self.feature_columns[0]])
    
    def get_current_atr(self) -> Optional[float]:
        """
        Get the current ATR (Average True Range) value.
        
        Returns raw ATR if 'atr' column exists, or calculates it from 'atr_rel'
        and close price if available. Returns None if ATR cannot be determined.
        
        Returns:
            Current ATR value or None
        """
        if self.current_step >= len(self.data):
            step = len(self.data) - 1
        else:
            step = self.current_step
        
        row = self.data.iloc[step]
        
        # Option 1: Raw ATR column exists
        if 'atr' in self.data.columns:
            return float(row['atr'])
        
        # Option 2: Calculate from atr_rel (atr_rel = log(ATR) - log(close))
        # So ATR = close * exp(atr_rel)
        if 'atr_rel' in self.data.columns:
            close = self._get_current_price()
            atr_rel = float(row['atr_rel'])
            return close * np.exp(atr_rel)
        
        return None
    
    def _is_terminated(self) -> bool:
        """
        Check if the episode should terminate.
        
        Returns:
            True if episode is done
        """
        # Episode ends when we've processed all data
        if self.current_step >= len(self.data):
            return True
        
        # Episode ends if capital is depleted
        if self.current_capital <= 0:
            return True
        
        return False
    
    def _position_to_dict(self, position: Optional[Position]) -> Optional[Dict[str, Any]]:
        """Convert position to dictionary for info."""
        if position is None:
            return None
        return {
            "id": position.id,
            "direction": position.direction,
            "entry_price": position.entry_price,
            "entry_time": position.entry_time.isoformat(),
            "size": position.size
        }
    
    def get_state(self) -> np.ndarray:
        """
        Get current observation state.
        
        Returns:
            Current observation array
        """
        return self._get_observation()
    
    def get_open_positions(self) -> List[Position]:
        """
        Return list of currently open positions (max 1).
        
        Returns:
            List containing current position or empty list
        """
        if self.current_position is not None:
            return [self.current_position]
        return []
    
    def has_open_position(self) -> bool:
        """
        Check if there is currently an open position.
        
        Returns:
            True if a position is open
        """
        return self.current_position is not None
    
    def can_open_position(self) -> bool:
        """
        Return True only if no position is currently open.
        
        This enforces the single position constraint.
        
        Returns:
            True if a new position can be opened
        """
        return not self.has_open_position()
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float
    ) -> float:
        """
        Calculate position size based on risk per trade and leverage.
        
        Args:
            entry_price: Planned entry price
            stop_loss_price: Stop loss price for risk calculation
            
        Returns:
            Position size (quantity)
        """
        if entry_price <= 0 or stop_loss_price <= 0:
            return 0.0
        
        # Risk per unit
        risk_per_unit = abs(entry_price - stop_loss_price)
        
        if risk_per_unit == 0:
            return 0.0
        
        # Risk amount in capital
        risk_amount = self.current_capital * self.risk_per_trade
        
        # Position size based on risk
        position_size = (risk_amount * self.leverage) / risk_per_unit
        
        return position_size
    
    def get_portfolio_value(self) -> float:
        """
        Get total portfolio value including unrealized P&L.
        
        Returns:
            Total portfolio value
        """
        current_price = self._get_current_price()
        unrealized_pnl = self.position_manager.get_unrealized_pnl(current_price)
        return self.current_capital + unrealized_pnl
    
    def render(self, mode: str = "human") -> None:
        """Render the environment state."""
        print(f"Step: {self.current_step}")
        print(f"Capital: {self.current_capital:.2f}")
        print(f"Position: {self.current_position}")
        print(f"Cumulative Reward: {self.cumulative_reward:.4f}")
