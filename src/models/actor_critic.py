"""Actor-Critic module combining Actor and Critic networks.

This module implements the combined Actor-Critic architecture for PPO training,
including coordinated inference and Generalized Advantage Estimation (GAE).
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from src.models.data_models import ActorNetworkConfig, CriticNetworkConfig, ActorCriticConfig
from src.models.actor_network import ActorNetwork
from src.models.critic_network import CriticNetwork


class ActorCritic(nn.Module):
    """Combined Actor-Critic module for PPO training.
    
    Combines ActorNetwork and CriticNetwork for coordinated training and inference.
    Implements Generalized Advantage Estimation (GAE) for advantage computation.
    
    Requirements: 6.8, 6.9
    """
    
    def __init__(
        self, 
        actor_config: ActorNetworkConfig, 
        critic_config: CriticNetworkConfig,
        gae_lambda: float = 0.95
    ):
        """Initialize with separate actor and critic configurations.
        
        Args:
            actor_config: Configuration for the actor network
            critic_config: Configuration for the critic network
            gae_lambda: Lambda parameter for GAE (default: 0.95)
        """
        super().__init__()
        self.actor = ActorNetwork(actor_config)
        self.critic = CriticNetwork(critic_config)
        self.gae_lambda = gae_lambda
        self._actor_config = actor_config
        self._critic_config = critic_config
    
    @classmethod
    def from_config(cls, config: ActorCriticConfig) -> 'ActorCritic':
        """Create ActorCritic from ActorCriticConfig.
        
        Args:
            config: ActorCriticConfig containing actor and critic configs
            
        Returns:
            ActorCritic instance
        """
        return cls(config.actor, config.critic, config.gae_lambda)

    def forward(
        self, 
        x: torch.Tensor, 
        actor_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        critic_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple, Tuple]:
        """Forward pass returning action logits, value estimate, and hidden states.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size) or (batch, input_size)
            actor_hidden: Optional hidden state for actor LSTM
            critic_hidden: Optional hidden state for critic LSTM
            
        Returns:
            Tuple of (action_logits, value, actor_hidden, critic_hidden)
        """
        action_logits, actor_hidden = self.actor(x, actor_hidden)
        value, critic_hidden = self.critic(x, critic_hidden)
        return action_logits, value, actor_hidden, critic_hidden
    
    def get_action_and_value(
        self, 
        x: torch.Tensor,
        actor_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        critic_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple, Tuple]:
        """Get action probabilities and state value for a given state.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size) or (batch, input_size)
            actor_hidden: Optional hidden state for actor LSTM
            critic_hidden: Optional hidden state for critic LSTM
            
        Returns:
            Tuple of (action_probs, value, actor_hidden, critic_hidden)
            - action_probs: Shape (batch, num_actions), probabilities summing to 1
            - value: Shape (batch,), scalar value estimates
        """
        action_probs, actor_hidden = self.actor.get_action_probs(x, actor_hidden)
        value, critic_hidden = self.critic.get_value(x, critic_hidden)
        return action_probs, value, actor_hidden, critic_hidden
    
    def compute_advantage(
        self, 
        rewards: torch.Tensor, 
        values: torch.Tensor, 
        next_values: torch.Tensor, 
        dones: torch.Tensor, 
        gamma: float
    ) -> torch.Tensor:
        """Compute Generalized Advantage Estimation (GAE) for PPO updates.
        
        GAE formula: A_t = sum(gamma^i * lambda^i * delta_{t+i})
        where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        
        Args:
            rewards: Tensor of rewards, shape (batch,) or (seq_len,)
            values: Tensor of value estimates V(s_t), shape (batch,) or (seq_len,)
            next_values: Tensor of next value estimates V(s_{t+1}), shape (batch,) or (seq_len,)
            dones: Tensor of done flags (1 if terminal, 0 otherwise), shape (batch,) or (seq_len,)
            gamma: Discount factor
            
        Returns:
            Tensor of advantages, same shape as rewards
            
        Requirements: 6.9
        """
        # Ensure tensors are float
        rewards = rewards.float()
        values = values.float()
        next_values = next_values.float()
        dones = dones.float()
        
        # Compute TD errors (deltas)
        # delta_t = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
        deltas = rewards + gamma * next_values * (1 - dones) - values
        
        # Compute GAE advantages
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        
        # Iterate backwards through time
        for t in reversed(range(len(rewards))):
            # GAE: A_t = delta_t + gamma * lambda * (1 - done) * A_{t+1}
            gae = deltas[t] + gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        return advantages

    def save(self, path: str) -> None:
        """Save both actor and critic weights and architectures (Requirement 6.8).
        
        Args:
            path: File path to save the model
        """
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_config': {
                'input_size': self._actor_config.input_size,
                'hidden_sizes': self._actor_config.hidden_sizes,
                'lstm_hidden_size': self._actor_config.lstm_hidden_size,
                'lstm_num_layers': self._actor_config.lstm_num_layers,
                'num_actions': self._actor_config.num_actions,
            },
            'critic_config': {
                'input_size': self._critic_config.input_size,
                'hidden_sizes': self._critic_config.hidden_sizes,
                'lstm_hidden_size': self._critic_config.lstm_hidden_size,
                'lstm_num_layers': self._critic_config.lstm_num_layers,
            },
            'gae_lambda': self.gae_lambda,
        }
        torch.save(checkpoint, path)
    
    @classmethod
    def load(cls, path: str) -> 'ActorCritic':
        """Load both actor and critic weights and architectures (Requirement 6.8).
        
        Args:
            path: File path to load the model from
            
        Returns:
            ActorCritic instance with loaded weights
        """
        checkpoint = torch.load(path, weights_only=False)
        
        actor_config = ActorNetworkConfig(**checkpoint['actor_config'])
        critic_config = CriticNetworkConfig(**checkpoint['critic_config'])
        gae_lambda = checkpoint.get('gae_lambda', 0.95)
        
        module = cls(actor_config, critic_config, gae_lambda)
        module.actor.load_state_dict(checkpoint['actor_state_dict'])
        module.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        return module
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """Get information about the combined architecture.
        
        Returns:
            Dictionary with architecture details for both networks
        """
        return {
            'actor': self.actor.get_architecture_info(),
            'critic': self.critic.get_architecture_info(),
            'gae_lambda': self.gae_lambda,
            'total_parameters': sum(p.numel() for p in self.parameters()),
        }
