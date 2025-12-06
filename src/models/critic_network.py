"""Critic Network for value estimation in Actor-Critic architecture.

This module implements the Critic network that outputs state value estimates
for advantage computation in PPO training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from src.models.data_models import CriticNetworkConfig


class CriticNetwork(nn.Module):
    """Neural network that estimates state values for advantage computation.
    
    Architecture:
    - LSTM layers for temporal pattern capture
    - Configurable fully-connected hidden layers
    - Value head outputting single scalar state value estimate
    
    Requirements: 6.2, 6.4, 6.5, 6.7, 6.8
    """
    
    def __init__(self, config: CriticNetworkConfig):
        """Initialize critic network with configuration.
        
        Args:
            config: CriticNetworkConfig with network architecture parameters
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
        
        # Fully-connected hidden layers (Requirements 6.2, 6.4)
        self.fc_layers = nn.ModuleList()
        layer_sizes = [config.lstm_hidden_size] + config.hidden_sizes
        for i in range(len(layer_sizes) - 1):
            self.fc_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        # Value head outputting single scalar (Requirement 6.7)
        if config.hidden_sizes:
            self.value_head = nn.Linear(config.hidden_sizes[-1], 1)
        else:
            self.value_head = nn.Linear(config.lstm_hidden_size, 1)
    
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass returning state value estimate and hidden state.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            hidden: Optional tuple of (h_0, c_0) for LSTM
            
        Returns:
            Tuple of (value, (h_n, c_n))
            - value: Shape (batch, 1) - scalar value estimate
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
        
        # Value head outputs single scalar
        value = self.value_head(out)
        
        return value, hidden
    
    def get_value(
        self, 
        x: torch.Tensor, 
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Get state value estimate as a scalar (Requirement 6.7).
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size) or (batch, input_size)
            hidden: Optional tuple of (h_0, c_0) for LSTM
            
        Returns:
            Tuple of (value, hidden)
            - value: Shape (batch,) - scalar value estimate squeezed
            - hidden: Tuple of hidden and cell states
        """
        value, hidden = self.forward(x, hidden)
        return value.squeeze(-1), hidden
    
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
            }
        }
        torch.save(checkpoint, path)
    
    @classmethod
    def load(cls, path: str) -> 'CriticNetwork':
        """Load network weights and architecture (Requirement 6.8).
        
        Args:
            path: File path to load the model from
            
        Returns:
            CriticNetwork instance with loaded weights
        """
        checkpoint = torch.load(path, weights_only=False)
        config = CriticNetworkConfig(**checkpoint['config'])
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
            'output_size': 1,  # Always scalar
            'total_parameters': sum(p.numel() for p in self.parameters()),
        }
