"""Actor Network for policy decisions in Actor-Critic architecture.

This module implements the Actor network that outputs action probabilities
for the Order Opening Agent (3 actions) and Order Closing Agent (2 actions).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from src.models.data_models import ActorNetworkConfig


class ActorNetwork(nn.Module):
    """Neural network that outputs action probabilities for policy decisions.
    
    Architecture:
    - LSTM layers for temporal pattern capture
    - Configurable fully-connected hidden layers
    - Action head with softmax for action probabilities
    
    Requirements: 6.1, 6.3, 6.5, 6.6, 6.8
    """
    
    def __init__(self, config: ActorNetworkConfig):
        """Initialize actor network with configuration.
        
        Args:
            config: ActorNetworkConfig with network architecture parameters
        """
        super().__init__()
        self.config = config
        
        # LSTM layers for temporal dependencies (Requirement 6.5)
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            batch_first=True
        )
        
        # Fully-connected hidden layers (Requirements 6.1, 6.3)
        self.fc_layers = nn.ModuleList()
        layer_sizes = [config.lstm_hidden_size] + config.hidden_sizes
        for i in range(len(layer_sizes) - 1):
            self.fc_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        # Action head outputting action logits (Requirement 6.6)
        if config.hidden_sizes:
            self.action_head = nn.Linear(config.hidden_sizes[-1], config.num_actions)
        else:
            self.action_head = nn.Linear(config.lstm_hidden_size, config.num_actions)
    
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass returning action logits and hidden state.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            hidden: Optional tuple of (h_0, c_0) for LSTM
            
        Returns:
            Tuple of (action_logits, (h_n, c_n))
            - action_logits: Shape (batch, num_actions)
            - hidden: Tuple of hidden and cell states
        """
        # Handle 2D input (batch, features) by adding sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)
        
        # LSTM forward pass
        if hidden is None:
            lstm_out, hidden = self.lstm(x)
        else:
            lstm_out, hidden = self.lstm(x, hidden)
        
        # Take the last output from LSTM sequence
        out = lstm_out[:, -1, :]  # (batch, lstm_hidden_size)
        
        # Pass through fully-connected layers with ReLU activation
        for fc in self.fc_layers:
            out = F.relu(fc(out))
        
        # Action head outputs logits
        action_logits = self.action_head(out)
        
        return action_logits, hidden
    
    def get_action_probs(
        self, 
        x: torch.Tensor, 
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Get action probabilities via softmax activation (Requirement 6.6).
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size) or (batch, input_size)
            hidden: Optional tuple of (h_0, c_0) for LSTM
            
        Returns:
            Tuple of (action_probs, hidden)
            - action_probs: Shape (batch, num_actions), probabilities summing to 1
            - hidden: Tuple of hidden and cell states
        """
        action_logits, hidden = self.forward(x, hidden)
        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs, hidden
    
    def save(self, path: str) -> None:
        """Save network weights and architecture (Requirement 6.8).
        
        Args:
            path: File path to save the model
        """
        checkpoint = {
            'state_dict': self.state_dict(),
            'config': {
                'input_size': self.config.input_size,
                'hidden_sizes': self.config.hidden_sizes,
                'lstm_hidden_size': self.config.lstm_hidden_size,
                'lstm_num_layers': self.config.lstm_num_layers,
                'num_actions': self.config.num_actions,
            }
        }
        torch.save(checkpoint, path)
    
    @classmethod
    def load(cls, path: str) -> 'ActorNetwork':
        """Load network weights and architecture (Requirement 6.8).
        
        Args:
            path: File path to load the model from
            
        Returns:
            ActorNetwork instance with loaded weights
        """
        checkpoint = torch.load(path, weights_only=False)
        config = ActorNetworkConfig(**checkpoint['config'])
        network = cls(config)
        network.load_state_dict(checkpoint['state_dict'])
        return network
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """Get information about the network architecture.
        
        Returns:
            Dictionary with architecture details
        """
        return {
            'input_size': self.config.input_size,
            'lstm_hidden_size': self.config.lstm_hidden_size,
            'lstm_num_layers': self.config.lstm_num_layers,
            'hidden_sizes': self.config.hidden_sizes,
            'num_actions': self.config.num_actions,
            'total_parameters': sum(p.numel() for p in self.parameters()),
        }
