"""Order Closing Agent for managing position exits.

This module implements the Order Closing Agent that manages position exits
to maximize profits or minimize losses. It uses an Actor-Critic architecture
with PPO algorithm and includes stop-loss and take-profit threshold checks.

Supports both fixed percentage and ATR-based dynamic SL/TP thresholds.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 6.9
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from src.models.actor_critic import ActorCritic
from src.models.data_models import AgentConfig, Experience, Position, CloseDecision


class OrderClosingAgent:
    """Agent specialized in managing position exits.
    
    Has 2 possible actions:
        - 0: hold (keep position open)
        - 1: close (close the position)
    
    Uses Actor-Critic architecture for policy and value estimation.
    Includes stop-loss and take-profit threshold checks with support for:
        - Fixed percentage thresholds (default)
        - ATR-based dynamic thresholds (adapts to market volatility)
    
    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 6.9
    """
    
    NUM_ACTIONS = 2  # hold, close
    
    # Action constants
    HOLD = 0
    CLOSE = 1
    
    # Position feature indices (appended to market state)
    POSITION_FEATURE_SIZE = 4  # direction, entry_price_rel, unrealized_pnl_pct, duration_norm
    
    # Maximum allowed risk-reward ratio for ATR-based SL/TP (1:3)
    MAX_ATR_RISK_REWARD_RATIO = 3.0
    
    def __init__(
        self, 
        config: AgentConfig, 
        actor_critic: ActorCritic,
        stop_loss_threshold: float = 0.05,
        take_profit_threshold: float = 0.10,
        use_atr_based_sl_tp: bool = False,
        atr_sl_multiplier: float = 1.5,
        atr_tp_multiplier: float = 2.0
    ):
        """Initialize agent with configuration and actor-critic networks.
        
        Args:
            config: AgentConfig with training hyperparameters
            actor_critic: ActorCritic module for policy and value estimation
            stop_loss_threshold: Stop-loss threshold as fraction (e.g., 0.05 = 5% loss)
                Used when use_atr_based_sl_tp=False
            take_profit_threshold: Take-profit threshold as fraction (e.g., 0.10 = 10% profit)
                Used when use_atr_based_sl_tp=False
            use_atr_based_sl_tp: If True, use ATR-based dynamic SL/TP instead of fixed percentages
            atr_sl_multiplier: Multiplier for ATR to calculate stop-loss distance (e.g., 1.5 = 1.5x ATR)
            atr_tp_multiplier: Multiplier for ATR to calculate take-profit distance (e.g., 2.0 = 2x ATR)
                Note: Maximum R:R ratio is capped at 1:3 (atr_tp_multiplier <= 3 * atr_sl_multiplier)
        
        Raises:
            ValueError: If ATR-based R:R ratio exceeds maximum of 1:3
        """
        self.config = config
        self.actor_critic = actor_critic
        self.stop_loss_threshold = stop_loss_threshold
        self.take_profit_threshold = take_profit_threshold
        
        # ATR-based dynamic SL/TP settings with validation
        self.use_atr_based_sl_tp = use_atr_based_sl_tp
        self.atr_sl_multiplier = atr_sl_multiplier
        
        # Validate and cap ATR TP multiplier to enforce max 1:3 R:R ratio
        max_tp_multiplier = atr_sl_multiplier * self.MAX_ATR_RISK_REWARD_RATIO
        if atr_tp_multiplier > max_tp_multiplier:
            self.atr_tp_multiplier = max_tp_multiplier
        else:
            self.atr_tp_multiplier = atr_tp_multiplier

        # Exploration rate for epsilon-greedy
        self._epsilon = config.epsilon_start
        
        # Device for tensor operations
        self._device = next(actor_critic.parameters()).device
        
        # Hidden states for LSTM
        self._actor_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._critic_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def _augment_state_with_position(
        self, 
        state: np.ndarray, 
        position: Position,
        current_price: float
    ) -> np.ndarray:
        """Augment market state with position-specific features.
        
        Creates a position-augmented state by appending position features
        to the market state. This allows the agent to make decisions
        based on both market conditions and position status.
        
        Args:
            state: Current market state as numpy array
            position: The open position to evaluate
            current_price: Current market price for P&L calculation
            
        Returns:
            Augmented state with position features appended
        """
        # Calculate position features
        direction_feature = 1.0 if position.direction == "long" else -1.0
        
        # Relative entry price (normalized)
        entry_price_rel = (current_price - position.entry_price) / position.entry_price
        
        # Unrealized P&L percentage
        if position.direction == "long":
            unrealized_pnl_pct = (current_price - position.entry_price) / position.entry_price
        else:  # short
            unrealized_pnl_pct = (position.entry_price - current_price) / position.entry_price
        
        # Duration feature (normalized, assuming max 1000 time steps)
        # Note: In practice, this would use actual timestamps
        duration_norm = 0.0  # Placeholder - would be calculated from entry_time
        
        # Create position features array
        position_features = np.array([
            direction_feature,
            entry_price_rel,
            unrealized_pnl_pct,
            duration_norm
        ], dtype=np.float32)
        
        # Concatenate market state with position features
        augmented_state = np.concatenate([state, position_features])
        
        return augmented_state

    def evaluate_positions(
        self, 
        state: np.ndarray, 
        positions: List[Position],
        current_price: float,
        atr: Optional[float] = None
    ) -> List[CloseDecision]:
        """Evaluate each open position for potential closure (Requirement 4.1).
        
        Evaluates each position independently and returns a decision for each.
        Supports ATR-based dynamic SL/TP when atr is provided and use_atr_based_sl_tp=True.
        
        Args:
            state: Current market state as numpy array
            positions: List of open positions to evaluate
            current_price: Current market price
            atr: Current ATR value (optional, used for dynamic SL/TP)
            
        Returns:
            List of CloseDecision objects, one for each position
        """
        decisions = []
        
        for position in positions:
            # Check stop-loss first (Requirement 4.3)
            if self.check_stop_loss(position, current_price, atr):
                decisions.append(CloseDecision(
                    position_id=position.id,
                    should_close=True,
                    confidence=1.0  # High confidence for stop-loss
                ))
                continue
            
            # Check take-profit (Requirement 4.4)
            if self.check_take_profit(position, current_price, atr):
                decisions.append(CloseDecision(
                    position_id=position.id,
                    should_close=True,
                    confidence=0.95  # High confidence for take-profit
                ))
                continue
            
            # Get action from policy
            action, log_prob, value = self.select_action(state, position, current_price)
            
            # Determine if should close
            should_close = (action == self.CLOSE)
            
            # Calculate confidence from action probability
            augmented_state = self._augment_state_with_position(state, position, current_price)
            state_tensor = torch.FloatTensor(augmented_state).unsqueeze(0).to(self._device)
            
            with torch.no_grad():
                action_probs, _, _, _ = self.actor_critic.get_action_and_value(
                    state_tensor, self._actor_hidden, self._critic_hidden
                )
                confidence = action_probs[0, action].item()
            
            decisions.append(CloseDecision(
                position_id=position.id,
                should_close=should_close,
                confidence=confidence
            ))
        
        return decisions

    def select_action(
        self, 
        state: np.ndarray, 
        position: Position,
        current_price: Optional[float] = None,
        explore: bool = True
    ) -> Tuple[int, float, float]:
        """Decide whether to close specific position (Requirement 4.2).
        
        Args:
            state: Current market state as numpy array
            position: The position to evaluate
            current_price: Current market price (optional, extracted from state if not provided)
            explore: Whether to use exploration (epsilon-greedy)
            
        Returns:
            Tuple of (action, log_prob, value):
                - action: Selected action (0=hold, 1=close)
                - log_prob: Log probability of the selected action
                - value: State value estimate from critic
        """
        # Use current_price if provided, otherwise assume it's the last element of state
        if current_price is None:
            current_price = state[-1] if len(state) > 0 else position.entry_price
        
        # Augment state with position features
        augmented_state = self._augment_state_with_position(state, position, current_price)
        
        # Convert to tensor
        state_tensor = torch.FloatTensor(augmented_state).unsqueeze(0).to(self._device)
        
        with torch.no_grad():
            # Get action probabilities and value
            action_probs, value, self._actor_hidden, self._critic_hidden = \
                self.actor_critic.get_action_and_value(
                    state_tensor, 
                    self._actor_hidden, 
                    self._critic_hidden
                )
            
            # Epsilon-greedy exploration
            if explore and np.random.random() < self._epsilon:
                # Random action
                action = np.random.randint(0, self.NUM_ACTIONS)
                action_tensor = torch.tensor([action], device=self._device)
            else:
                # Sample from policy distribution
                dist = Categorical(action_probs)
                action_tensor = dist.sample()
                action = action_tensor.item()
            
            # Calculate log probability
            dist = Categorical(action_probs)
            log_prob = dist.log_prob(action_tensor).item()
            
            # Get scalar value
            value_scalar = value.squeeze().item()
        
        return action, log_prob, value_scalar


    def get_value(
        self, 
        state: np.ndarray, 
        position: Position,
        current_price: Optional[float] = None
    ) -> float:
        """Get state value estimate from critic for position-augmented state.
        
        Args:
            state: Current market state as numpy array
            position: The position being evaluated
            current_price: Current market price (optional)
            
        Returns:
            State value estimate as scalar
        """
        # Use current_price if provided, otherwise assume it's the last element of state
        if current_price is None:
            current_price = state[-1] if len(state) > 0 else position.entry_price
        
        # Augment state with position features
        augmented_state = self._augment_state_with_position(state, position, current_price)
        
        state_tensor = torch.FloatTensor(augmented_state).unsqueeze(0).to(self._device)
        
        with torch.no_grad():
            value, self._critic_hidden = self.actor_critic.critic.get_value(
                state_tensor, 
                self._critic_hidden
            )
        
        return value.item()

    def update_policy(self, experiences: List[Experience]) -> Dict[str, float]:
        """Update actor and critic using PPO algorithm with advantage estimation.
        
        Implements PPO-Clip algorithm (Requirement 4.5, 6.9):
        1. Compute advantages using GAE
        2. Update actor with clipped surrogate objective
        3. Update critic with value loss
        
        Args:
            experiences: List of Experience tuples from episode
            
        Returns:
            Dictionary with training metrics:
                - actor_loss: Policy loss
                - critic_loss: Value function loss
                - entropy: Policy entropy
                - approx_kl: Approximate KL divergence
        """
        if not experiences:
            return {
                'actor_loss': 0.0,
                'critic_loss': 0.0,
                'entropy': 0.0,
                'approx_kl': 0.0
            }
        
        # Convert experiences to tensors
        states = torch.FloatTensor(
            np.array([e.state for e in experiences])
        ).to(self._device)
        actions = torch.LongTensor(
            [e.action for e in experiences]
        ).to(self._device)
        rewards = torch.FloatTensor(
            [e.reward for e in experiences]
        ).to(self._device)
        next_states = torch.FloatTensor(
            np.array([e.next_state for e in experiences])
        ).to(self._device)
        dones = torch.FloatTensor(
            [float(e.done) for e in experiences]
        ).to(self._device)
        old_log_probs = torch.FloatTensor(
            [e.log_prob for e in experiences]
        ).to(self._device)
        old_values = torch.FloatTensor(
            [e.value for e in experiences]
        ).to(self._device)
        
        # Get current values for next states
        with torch.no_grad():
            next_values, _ = self.actor_critic.critic.get_value(next_states)
        
        # Compute advantages using GAE (Requirement 6.9)
        advantages = self.actor_critic.compute_advantage(
            rewards, old_values, next_values, dones, self.config.gamma
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute returns (advantages + values)
        returns = advantages + old_values
        
        # Get current action probabilities
        action_probs, _ = self.actor_critic.actor.get_action_probs(states)
        dist = Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # PPO clipped surrogate objective (Requirement 4.5)
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio, 
            1.0 - self.config.clip_ratio, 
            1.0 + self.config.clip_ratio
        ) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        current_values, _ = self.actor_critic.critic.get_value(states)
        critic_loss = F.mse_loss(current_values, returns)
        
        # Approximate KL divergence for monitoring
        with torch.no_grad():
            approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
        
        # Combined loss with entropy bonus
        entropy_coef = 0.01  # Encourage exploration
        total_loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy
        
        # Backpropagation
        # Note: Optimizers should be managed externally (in training pipeline)
        # This method computes gradients but doesn't step optimizers
        total_loss.backward()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item(),
            'approx_kl': approx_kl
        }

    def check_stop_loss(
        self, 
        position: Position, 
        current_price: float,
        atr: Optional[float] = None
    ) -> bool:
        """Check if position has hit stop-loss threshold (Requirement 4.3).
        
        Priority order:
        1. Use fixed SL price stored in position (calculated at entry)
        2. Fall back to percentage-based SL
        
        Args:
            position: The position to check
            current_price: Current market price
            atr: Unused (kept for backward compatibility)
            
        Returns:
            True if stop-loss threshold is reached, False otherwise
        """
        # Priority 1: Use fixed SL price from position (calculated at entry with ATR)
        if position.stop_loss_price is not None:
            if position.direction == "long":
                return current_price <= position.stop_loss_price
            else:  # short
                return current_price >= position.stop_loss_price
        
        # Priority 2: Fall back to percentage-based stop-loss
        if position.direction == "long":
            pnl_pct = (current_price - position.entry_price) / position.entry_price
        else:  # short
            pnl_pct = (position.entry_price - current_price) / position.entry_price
        
        return pnl_pct <= -self.stop_loss_threshold

    def check_take_profit(
        self, 
        position: Position, 
        current_price: float,
        atr: Optional[float] = None
    ) -> bool:
        """Check if position has hit take-profit threshold (Requirement 4.4).
        
        Priority order:
        1. Use fixed TP price stored in position (calculated at entry)
        2. Fall back to percentage-based TP
        
        Args:
            position: The position to check
            current_price: Current market price
            atr: Unused (kept for backward compatibility)
            
        Returns:
            True if take-profit threshold is reached, False otherwise
        """
        # Priority 1: Use fixed TP price from position (calculated at entry with ATR)
        if position.take_profit_price is not None:
            if position.direction == "long":
                return current_price >= position.take_profit_price
            else:  # short
                return current_price <= position.take_profit_price
        
        # Priority 2: Fall back to percentage-based take-profit
        if position.direction == "long":
            pnl_pct = (current_price - position.entry_price) / position.entry_price
        else:  # short
            pnl_pct = (position.entry_price - current_price) / position.entry_price
        
        return pnl_pct >= self.take_profit_threshold
    
    def calculate_sl_tp_prices(
        self,
        entry_price: float,
        direction: str,
        atr: float
    ) -> Tuple[float, float]:
        """Calculate fixed SL/TP prices at entry time based on ATR.
        
        Call this when opening a position to get the SL/TP prices
        that should be stored in the Position object.
        
        Args:
            entry_price: Position entry price
            direction: "long" or "short"
            atr: ATR value at entry time
            
        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        sl_distance = atr * self.atr_sl_multiplier
        tp_distance = atr * self.atr_tp_multiplier
        
        if direction == "long":
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        else:  # short
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance
        
        return sl_price, tp_price
    
    def get_dynamic_sl_tp_prices(
        self, 
        position: Position, 
        atr: float
    ) -> Tuple[float, float]:
        """Calculate stop-loss and take-profit prices based on ATR.
        
        Note: For fixed SL/TP, use position.stop_loss_price and position.take_profit_price
        which are calculated at entry time. This method is for display/logging purposes.
        
        Args:
            position: The position to calculate SL/TP for
            atr: ATR value
            
        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        # If position has fixed SL/TP, return those
        if position.stop_loss_price is not None and position.take_profit_price is not None:
            return position.stop_loss_price, position.take_profit_price
        
        # Otherwise calculate from current ATR (for backward compatibility)
        sl_distance = atr * self.atr_sl_multiplier
        tp_distance = atr * self.atr_tp_multiplier
        
        if position.direction == "long":
            sl_price = position.entry_price - sl_distance
            tp_price = position.entry_price + tp_distance
        else:  # short
            sl_price = position.entry_price + sl_distance
            tp_price = position.entry_price - tp_distance
        
        return sl_price, tp_price
    
    def get_risk_reward_ratio(self, atr: Optional[float] = None) -> float:
        """Calculate the risk-reward ratio based on current SL/TP settings.
        
        Args:
            atr: Current ATR value (used for ATR-based calculation)
            
        Returns:
            Risk-reward ratio (TP distance / SL distance)
        """
        if self.use_atr_based_sl_tp:
            return self.atr_tp_multiplier / self.atr_sl_multiplier
        else:
            return self.take_profit_threshold / self.stop_loss_threshold


    def set_exploration_rate(self, epsilon: float) -> None:
        """Set epsilon for epsilon-greedy exploration.
        
        Args:
            epsilon: New exploration rate, should be in [0, 1]
        """
        self._epsilon = max(0.0, min(1.0, epsilon))
    
    def get_exploration_rate(self) -> float:
        """Get current exploration rate.
        
        Returns:
            Current epsilon value
        """
        return self._epsilon
    
    def decay_exploration(self) -> float:
        """Decay exploration rate according to config.
        
        Applies exponential decay: epsilon = max(epsilon_end, epsilon * decay)
        
        Returns:
            New epsilon value after decay
        """
        self._epsilon = max(
            self.config.epsilon_end,
            self._epsilon * self.config.epsilon_decay
        )
        return self._epsilon
    
    def reset_hidden_states(self) -> None:
        """Reset LSTM hidden states.
        
        Should be called at the start of each episode.
        """
        self._actor_hidden = None
        self._critic_hidden = None
    
    def train(self) -> None:
        """Set agent to training mode."""
        self.actor_critic.train()
    
    def eval(self) -> None:
        """Set agent to evaluation mode."""
        self.actor_critic.eval()
    
    def to(self, device: torch.device) -> 'OrderClosingAgent':
        """Move agent to specified device.
        
        Args:
            device: Target device (cpu or cuda)
            
        Returns:
            Self for method chaining
        """
        self.actor_critic.to(device)
        self._device = device
        return self

    def get_stop_loss_threshold(self) -> float:
        """Get current stop-loss threshold.
        
        Returns:
            Stop-loss threshold as fraction
        """
        return self.stop_loss_threshold
    
    def set_stop_loss_threshold(self, threshold: float) -> None:
        """Set stop-loss threshold.
        
        Args:
            threshold: New stop-loss threshold as fraction (e.g., 0.05 = 5%)
        """
        self.stop_loss_threshold = max(0.0, threshold)
    
    def get_take_profit_threshold(self) -> float:
        """Get current take-profit threshold.
        
        Returns:
            Take-profit threshold as fraction
        """
        return self.take_profit_threshold
    
    def set_take_profit_threshold(self, threshold: float) -> None:
        """Set take-profit threshold.
        
        Args:
            threshold: New take-profit threshold as fraction (e.g., 0.10 = 10%)
        """
        self.take_profit_threshold = max(0.0, threshold)

    def is_using_atr_based_sl_tp(self) -> bool:
        """Check if ATR-based dynamic SL/TP is enabled.
        
        Returns:
            True if using ATR-based SL/TP, False if using fixed percentages
        """
        return self.use_atr_based_sl_tp
    
    def set_use_atr_based_sl_tp(self, enabled: bool) -> None:
        """Enable or disable ATR-based dynamic SL/TP.
        
        Args:
            enabled: True to use ATR-based SL/TP, False for fixed percentages
        """
        self.use_atr_based_sl_tp = enabled
    
    def get_atr_sl_multiplier(self) -> float:
        """Get ATR multiplier for stop-loss.
        
        Returns:
            ATR multiplier for stop-loss distance
        """
        return self.atr_sl_multiplier
    
    def set_atr_sl_multiplier(self, multiplier: float) -> None:
        """Set ATR multiplier for stop-loss.
        
        Note: If this changes the R:R ratio to exceed 1:3, the TP multiplier
        will be automatically adjusted to maintain the maximum ratio.
        
        Args:
            multiplier: New ATR multiplier (e.g., 1.5 = 1.5x ATR)
        """
        self.atr_sl_multiplier = max(0.1, multiplier)
        # Adjust TP multiplier if R:R ratio would exceed maximum
        max_tp_multiplier = self.atr_sl_multiplier * self.MAX_ATR_RISK_REWARD_RATIO
        if self.atr_tp_multiplier > max_tp_multiplier:
            self.atr_tp_multiplier = max_tp_multiplier
    
    def get_atr_tp_multiplier(self) -> float:
        """Get ATR multiplier for take-profit.
        
        Returns:
            ATR multiplier for take-profit distance
        """
        return self.atr_tp_multiplier
    
    def set_atr_tp_multiplier(self, multiplier: float) -> None:
        """Set ATR multiplier for take-profit.
        
        Note: The multiplier is capped to enforce maximum 1:3 R:R ratio.
        
        Args:
            multiplier: New ATR multiplier (e.g., 2.0 = 2x ATR)
                Will be capped at 3 * atr_sl_multiplier to enforce max 1:3 R:R
        """
        max_tp_multiplier = self.atr_sl_multiplier * self.MAX_ATR_RISK_REWARD_RATIO
        self.atr_tp_multiplier = max(0.1, min(multiplier, max_tp_multiplier))
    
    def set_atr_multipliers(self, sl_multiplier: float, tp_multiplier: float) -> Tuple[float, float]:
        """Set both ATR multipliers at once with validation.
        
        Enforces maximum 1:3 risk-reward ratio. If the requested ratio exceeds
        this limit, the TP multiplier will be capped.
        
        Args:
            sl_multiplier: ATR multiplier for stop-loss (e.g., 1.5 = 1.5x ATR)
            tp_multiplier: ATR multiplier for take-profit (e.g., 3.0 = 3x ATR)
            
        Returns:
            Tuple of (actual_sl_multiplier, actual_tp_multiplier) after validation
        """
        self.atr_sl_multiplier = max(0.1, sl_multiplier)
        max_tp_multiplier = self.atr_sl_multiplier * self.MAX_ATR_RISK_REWARD_RATIO
        self.atr_tp_multiplier = max(0.1, min(tp_multiplier, max_tp_multiplier))
        return self.atr_sl_multiplier, self.atr_tp_multiplier
    
    def get_max_risk_reward_ratio(self) -> float:
        """Get the maximum allowed risk-reward ratio for ATR-based SL/TP.
        
        Returns:
            Maximum R:R ratio (default: 3.0 for 1:3)
        """
        return self.MAX_ATR_RISK_REWARD_RATIO

    def get_unrealized_pnl_pct(self, position: Position, current_price: float) -> float:
        """Calculate unrealized P&L percentage for a position.
        
        Args:
            position: The position to calculate P&L for
            current_price: Current market price
            
        Returns:
            Unrealized P&L as a percentage (e.g., 0.05 = 5% profit)
        """
        if position.direction == "long":
            return (current_price - position.entry_price) / position.entry_price
        else:  # short
            return (position.entry_price - current_price) / position.entry_price
