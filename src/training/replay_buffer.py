"""Replay buffer for storing and sampling experience tuples.

This module implements a replay buffer with configurable capacity for
storing experience tuples during training. It supports mini-batch sampling
for PPO policy updates.

Requirements: 7.2, 7.3
"""

from collections import deque
from typing import List, Optional
import random

from src.models.data_models import Experience


class ReplayBuffer:
    """Stores and samples experience tuples for training.
    
    The buffer uses a deque with a maximum capacity. When the buffer is full,
    the oldest experiences are automatically removed when new ones are added.
    
    Attributes:
        capacity: Maximum number of experiences to store.
        buffer: Internal deque storing Experience objects.
    """
    
    def __init__(self, capacity: int):
        """Initialize buffer with maximum capacity.
        
        Args:
            capacity: Maximum number of experiences to store. Must be positive.
            
        Raises:
            ValueError: If capacity is not a positive integer.
        """
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        
        self.capacity = capacity
        self._buffer: deque[Experience] = deque(maxlen=capacity)
    
    def push(self, experience: Experience) -> None:
        """Add experience to buffer.
        
        If the buffer is at capacity, the oldest experience is automatically
        removed to make room for the new one.
        
        Args:
            experience: Experience tuple to add to the buffer.
        """
        self._buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random mini-batch from buffer.
        
        Args:
            batch_size: Number of experiences to sample.
            
        Returns:
            List of randomly sampled Experience objects.
            
        Raises:
            ValueError: If batch_size is larger than buffer size or non-positive.
        """
        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {batch_size}")
        
        if batch_size > len(self._buffer):
            raise ValueError(
                f"Batch size ({batch_size}) cannot exceed buffer size ({len(self._buffer)})"
            )
        
        return random.sample(list(self._buffer), batch_size)
    
    def clear(self) -> None:
        """Clear all experiences from buffer."""
        self._buffer.clear()
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self._buffer)
    
    def is_full(self) -> bool:
        """Check if buffer is at capacity.
        
        Returns:
            True if buffer contains capacity number of experiences.
        """
        return len(self._buffer) >= self.capacity
    
    def get_all(self) -> List[Experience]:
        """Get all experiences in the buffer.
        
        Returns:
            List of all Experience objects in the buffer.
        """
        return list(self._buffer)
