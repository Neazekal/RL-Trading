"""Order Opening Agent for identifying entry opportunities.

This module implements the Order Opening Agent that identifies optimal entry points
for trading positions. It uses an Actor-Critic architecture with PPO algorithm
and integrates with the Signal Confirmation Layer for execution safety.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10, 6.9
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from src.models.actor_critic import ActorCritic
from src.models.data_models import AgentConfig, Experience
from src.agents.signal_confirmation import SignalConfirmationLayer


class OrderOpeningAgent:
    """Agent specialized in identifying entry opportunities.
    
    Has 3 possible actions:
        - 0: hold (no action)
        - 1: open_long
        - 2: open_short
    
    Uses Actor-Critic architecture with Signal Confirmation Layer for execution.
    
    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10, 6.9
    """
    
    NUM_ACTIONS = 3  # hold, open_long, open_short
    
    # Action constants
    HOLD = 0
    OPEN_LONG = 1
    OPEN_SHORT = 2
    
    def __init__(
        self, 
        config: AgentConfig, 
        actor_critic: ActorCritic,
        confirmation_layer: Optional[SignalConfirmationLayer] = None
    ):
        """Initialize agent with configuration, actor-critic networks, and confirmation layer.
        
        Args:
            config: AgentConfig with training hyperparameters
            actor_critic: ActorCritic module for policy and value estimation
            confirmation_layer: Optional SignalConfirmationLayer for signal filtering.
                If None, signals are executed immediately without confirmation.
        """
        self.config = config
        self.actor_critic = actor_critic
        self.confirmation_layer = confirmation_layer
        
        # Exploration rate for epsilon-greedy (Requirement 3.2)
        self._epsilon = config.epsilon_start
        
        # Device for tensor operations
        self._device = next(actor_critic.parameters()).device
        
        # Hidden states for LSTM
        self._actor_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._critic_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def select_action(
        self, 
        state: np.ndarray, 
        explore: bool = True
    ) -> Tuple[int, float, float]:
        """Select action using epsilon-greedy policy (Requirement 3.2).
        
        Args:
            state: Current market state as numpy array
            explore: Whether to use exploration (epsilon-greedy)
            
        Returns:
            Tuple of (action, log_prob, value):
                - action: Selected action (0=hold, 1=open_long, 2=open_short)
                - log_prob: Log probability of the selected action
                - value: State value estimate from critic
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self._device)
        
        with torch.no_grad():
            # Get action probabilities and value
            action_probs, value, self._actor_hidden, self._critic_hidden = \
                self.actor_critic.get_action_and_value(
                    state_tensor, 
                    self._actor_hidden, 
                    self._critic_hidden
                )
            
            # Epsilon-greedy exploration (Requirement 3.2)
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
    
    def get_confirmed_action(
        self, 
        state: np.ndarray, 
        explore: bool = True
    ) -> Tuple[int, float, float]:
        """Select action and pass through confirmation layer (Requirements 3.7-3.10).
        
        This method integrates the SignalConfirmationLayer to require N consecutive
        identical signals before executing an open action.
        
        Args:
            state: Current market state as numpy array
            explore: Whether to use exploration (epsilon-greedy)
            
        Returns:
            Tuple of (confirmed_action, log_prob, value):
                - confirmed_action: Action after confirmation layer processing
                - log_prob: Log probability of the raw action (before confirmation)
                - value: State value estimate from critic
        """
        # Get raw action from policy
        raw_action, log_prob, value = self.select_action(state, explore)
        
        # If no confirmation layer, return raw action
        if self.confirmation_layer is None:
            return raw_action, log_prob, value
        
        # Process through confirmation layer
        confirmed_action = self.confirmation_layer.process_signal(raw_action)
        
        return confirmed_action, log_prob, value
    
    def get_position_size(self, state: np.ndarray, balance: float) -> float:
        """Determine position size as percentage of balance (Requirement 3.3).
        
        Position size is bounded between 1% and 100% of available balance.
        
        Args:
            state: Current market state as numpy array
            balance: Available balance for trading
            
        Returns:
            Position size in absolute terms (between 0.01 * balance and 1.0 * balance)
        """
        # For now, use a simple heuristic based on state
        # In a more sophisticated implementation, this could be learned
        # or based on volatility/risk metrics from the state
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self._device)
        
        with torch.no_grad():
            # Get action probabilities to gauge confidence
            action_probs, _, self._actor_hidden, _ = \
                self.actor_critic.get_action_and_value(
                    state_tensor, 
                    self._actor_hidden, 
                    self._critic_hidden
                )
            
            # Use max probability as confidence measure
            confidence = action_probs.max().item()
            
            # Scale position size based on confidence
            # Min 1%, max 100% of balance (Requirement 3.3)
            min_size_pct = 0.01
            max_size_pct = 1.0
            
            # Linear scaling: higher confidence = larger position
            size_pct = min_size_pct + (max_size_pct - min_size_pct) * confidence
            
            # Clamp to valid range
            size_pct = max(min_size_pct, min(max_size_pct, size_pct))
        
        return size_pct * balance
    
    def get_value(self, state: np.ndarray) -> float:
        """Get state value estimate from critic network.
        
        Args:
            state: Current market state as numpy array
            
        Returns:
            State value estimate as scalar
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self._device)
        
        with torch.no_grad():
            value, self._critic_hidden = self.actor_critic.critic.get_value(
                state_tensor, 
                self._critic_hidden
            )
        
        return value.item()

    def update_policy(self, experiences: List[Experience]) -> Dict[str, float]:
        """Update actor and critic using PPO algorithm with advantage estimation.
        
        Implements PPO-Clip algorithm (Requirement 3.4, 6.9):
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
        
        # PPO clipped surrogate objective (Requirement 3.4)
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
    
    def set_exploration_rate(self, epsilon: float) -> None:
        """Set epsilon for epsilon-greedy exploration (Requirement 3.2).
        
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
    
    def reset_confirmation_layer(self) -> None:
        """Reset the signal confirmation layer.
        
        Should be called after a position is opened or at episode start.
        """
        if self.confirmation_layer is not None:
            self.confirmation_layer.reset()
    
    def train(self) -> None:
        """Set agent to training mode."""
        self.actor_critic.train()
    
    def eval(self) -> None:
        """Set agent to evaluation mode."""
        self.actor_critic.eval()
    
    def to(self, device: torch.device) -> 'OrderOpeningAgent':
        """Move agent to specified device.
        
        Args:
            device: Target device (cpu or cuda)
            
        Returns:
            Self for method chaining
        """
        self.actor_critic.to(device)
        self._device = device
        return self
    
    def get_confirmation_progress(self) -> Optional[Tuple[Optional[int], int, int]]:
        """Get current signal confirmation progress.
        
        Returns:
            Tuple of (current_signal, consecutive_count, required_confirmations)
            or None if no confirmation layer is configured
        """
        if self.confirmation_layer is None:
            return None
        return self.confirmation_layer.get_confirmation_progress()
