"""Reward Function for RL trading agents.

Implements reward calculation for both opening and closing agents,
following the specifications in Requirements 5.1-5.5.
"""

from typing import List

from src.models.data_models import RewardConfig, TradeResult


class RewardFunction:
    """
    Calculates rewards for agent actions.
    
    Reward rules (from Requirements 5.1-5.5):
    - Opening a position: small negative reward (open_penalty) to discourage excessive trading
    - Closing with profit P > 0: positive reward = P * profit_multiplier
    - Closing with loss L < 0: negative reward = L * loss_multiplier
    - Hold action: zero reward
    - Future rewards are discounted by discount_factor (gamma)
    """
    
    def __init__(self, config: RewardConfig):
        """
        Initialize the reward function with configuration.
        
        Args:
            config: RewardConfig containing:
                - open_penalty: Negative reward for opening positions
                - profit_multiplier: Multiplier for profitable trades
                - loss_multiplier: Multiplier for losing trades
                - discount_factor: Gamma for future reward discounting
        """
        self.config = config
        self.open_penalty = config.open_penalty
        self.profit_multiplier = config.profit_multiplier
        self.loss_multiplier = config.loss_multiplier
        self.discount_factor = config.discount_factor
    
    def calculate_open_reward(self, action: int) -> float:
        """
        Calculate reward for opening agent action.
        
        Args:
            action: Opening action (0=hold, 1=open_long, 2=open_short)
            
        Returns:
            Reward value:
                - 0.0 for hold action (action == 0)
                - negative open_penalty for open actions (action == 1 or 2)
        
        Requirements: 5.1, 5.4
        """
        # Hold action yields zero reward (Requirement 5.4)
        if action == 0:
            return 0.0
        
        # Opening a position yields negative reward (Requirement 5.1)
        # action 1 = open_long, action 2 = open_short
        if action in (1, 2):
            return -abs(self.open_penalty)  # Ensure it's negative
        
        # Invalid action, treat as hold
        return 0.0
    
    def calculate_close_reward(self, trade_result: TradeResult) -> float:
        """
        Calculate reward for closing agent based on trade P&L.
        
        Args:
            trade_result: TradeResult containing profit_loss and profit_loss_percent
            
        Returns:
            Reward value:
                - P * profit_multiplier if P > 0 (Requirement 5.2)
                - L * loss_multiplier if L < 0 (Requirement 5.3)
                - 0.0 if P == 0
        
        Requirements: 5.2, 5.3
        """
        pnl_percent = trade_result.profit_loss_percent
        
        if pnl_percent > 0:
            # Profitable trade: positive reward (Requirement 5.2)
            return pnl_percent * self.profit_multiplier
        elif pnl_percent < 0:
            # Losing trade: negative reward (Requirement 5.3)
            # Note: pnl_percent is already negative, so this produces negative reward
            return pnl_percent * self.loss_multiplier
        else:
            # Break-even trade
            return 0.0
    
    def calculate_hold_reward(self) -> float:
        """
        Calculate reward for hold action.
        
        Returns:
            0.0 (Requirement 5.4)
        
        Requirements: 5.4
        """
        return 0.0
    
    def apply_discount(self, rewards: List[float], gamma: float = None) -> List[float]:
        """
        Apply discount factor to future rewards.
        
        Computes discounted returns where the discounted return at step t equals:
        sum(gamma^i * r_{t+i}) for i from 0 to n-t
        
        Args:
            rewards: List of rewards [r0, r1, r2, ..., rn]
            gamma: Discount factor (uses config value if not provided)
            
        Returns:
            List of discounted returns at each step
        
        Requirements: 5.5
        """
        if gamma is None:
            gamma = self.discount_factor
        
        if not rewards:
            return []
        
        n = len(rewards)
        discounted_returns = [0.0] * n
        
        # Compute discounted return at each step
        # discounted_return[t] = sum(gamma^i * rewards[t+i]) for i from 0 to n-t-1
        for t in range(n):
            discounted_sum = 0.0
            for i in range(n - t):
                discounted_sum += (gamma ** i) * rewards[t + i]
            discounted_returns[t] = discounted_sum
        
        return discounted_returns
    
    def get_config(self) -> RewardConfig:
        """Return the current reward configuration."""
        return self.config
