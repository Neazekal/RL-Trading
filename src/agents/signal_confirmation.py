"""Signal Confirmation Layer for filtering noisy trading signals.

This module implements a safety layer that requires N consecutive identical
signals before executing a position. This reduces noise and confirms trading signals.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SignalConfirmationConfig:
    """Configuration for the Signal Confirmation Layer.
    
    Attributes:
        required_confirmations: Number of consecutive signals required (N).
            Default is 3.
        enabled: Whether the confirmation layer is active. Default is True.
    """
    required_confirmations: int = 3
    enabled: bool = True


class SignalConfirmationLayer:
    """Safety layer that requires N consecutive identical signals before execution.
    
    This layer filters noisy signals by requiring multiple consecutive identical
    open signals (open_long or open_short) before actually executing the position.
    
    Actions:
        0 = hold (no action)
        1 = open_long
        2 = open_short
    
    Behavior:
        - Hold actions (0) reset the counter
        - Open signals (1 or 2) increment the counter if they match the current signal
        - Different open signals reset the counter and start tracking the new signal
        - Only after N consecutive identical open signals is the action executed
        - Until N signals are reached, the output is hold (0)
    
    Example:
        With required_confirmations=3:
        Input sequence: [1, 1, 1] -> Output: [0, 0, 1]  (open_long on 3rd signal)
        Input sequence: [1, 1, 2] -> Output: [0, 0, 0]  (reset on different signal)
        Input sequence: [1, 0, 1] -> Output: [0, 0, 0]  (reset on hold)
    """
    
    # Action constants
    HOLD = 0
    OPEN_LONG = 1
    OPEN_SHORT = 2
    
    def __init__(self, required_confirmations: int = 3):
        """Initialize with number of required consecutive signals (N).
        
        Args:
            required_confirmations: Number of consecutive identical signals
                required before executing an open action. Must be >= 1.
        
        Raises:
            ValueError: If required_confirmations is less than 1.
        """
        if required_confirmations < 1:
            raise ValueError(
                f"required_confirmations must be >= 1, got {required_confirmations}"
            )
        
        self.required_confirmations = required_confirmations
        self._current_signal: Optional[int] = None  # 1=long, 2=short, None=no tracking
        self._consecutive_count: int = 0
    
    def process_signal(self, action: int) -> int:
        """Process agent action through confirmation layer.
        
        This method tracks consecutive signals and only outputs the actual
        open action after N consecutive identical signals are received.
        
        Args:
            action: Raw action from agent (0=hold, 1=open_long, 2=open_short)
        
        Returns:
            Confirmed action:
                - 0 (hold) if action is hold or if fewer than N consecutive signals
                - The original action (1 or 2) if N consecutive signals reached
        
        Raises:
            ValueError: If action is not 0, 1, or 2.
        """
        if action not in (self.HOLD, self.OPEN_LONG, self.OPEN_SHORT):
            raise ValueError(f"Invalid action: {action}. Must be 0, 1, or 2.")
        
        # Hold action resets the counter
        if action == self.HOLD:
            self.reset()
            return self.HOLD
        
        # Open signal (1 or 2)
        if self._current_signal is None:
            # Start tracking a new signal
            self._current_signal = action
            self._consecutive_count = 1
        elif self._current_signal == action:
            # Same signal, increment counter
            self._consecutive_count += 1
        else:
            # Different signal, reset and start tracking new one
            self._current_signal = action
            self._consecutive_count = 1
        
        # Check if we've reached the required confirmations
        if self._consecutive_count >= self.required_confirmations:
            confirmed_action = self._current_signal
            self.reset()  # Reset after execution
            return confirmed_action
        
        # Not enough confirmations yet, output hold
        return self.HOLD
    
    def reset(self) -> None:
        """Reset signal counter after position is opened or on sequence break.
        
        This clears the current signal tracking and resets the consecutive
        count to zero.
        """
        self._current_signal = None
        self._consecutive_count = 0
    
    def get_confirmation_progress(self) -> Tuple[Optional[int], int, int]:
        """Return current confirmation progress.
        
        Returns:
            A tuple of (current_signal, consecutive_count, required_confirmations):
                - current_signal: The signal being tracked (1, 2, or None)
                - consecutive_count: Number of consecutive signals received
                - required_confirmations: Total signals needed for confirmation
        """
        return (
            self._current_signal,
            self._consecutive_count,
            self.required_confirmations
        )
    
    @property
    def current_signal(self) -> Optional[int]:
        """Get the current signal being tracked."""
        return self._current_signal
    
    @property
    def consecutive_count(self) -> int:
        """Get the current consecutive signal count."""
        return self._consecutive_count
