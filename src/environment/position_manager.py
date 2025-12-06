"""Position Manager for tracking and managing trading positions."""

import uuid
from datetime import datetime
from typing import Dict, Optional

from src.models.data_models import Position, TradeResult


class PositionManager:
    """
    Manages open positions and calculates P&L.
    
    Handles position lifecycle: opening, tracking, and closing positions
    with proper fee calculation.
    """
    
    def __init__(self, trading_fee: float = 0.0004):
        """
        Initialize the PositionManager.
        
        Args:
            trading_fee: Fee rate per trade (e.g., 0.0004 = 0.04%)
        """
        self.trading_fee = trading_fee
        self._positions: Dict[str, Position] = {}
    
    def open_position(
        self,
        direction: str,
        price: float,
        size: float,
        entry_time: Optional[datetime] = None
    ) -> Position:
        """
        Open a new position.
        
        Args:
            direction: Position direction ("long" or "short")
            price: Entry price
            size: Position size (quantity)
            entry_time: Entry timestamp (defaults to now)
            
        Returns:
            The created Position object
            
        Raises:
            ValueError: If direction is invalid or price/size are non-positive
        """
        if direction not in ("long", "short"):
            raise ValueError(f"Invalid direction: {direction}. Must be 'long' or 'short'")
        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}")
        if size <= 0:
            raise ValueError(f"Size must be positive, got {size}")
        
        position_id = str(uuid.uuid4())
        position = Position(
            id=position_id,
            direction=direction,
            entry_price=price,
            entry_time=entry_time or datetime.now(),
            size=size
        )
        self._positions[position_id] = position
        return position
    
    def close_position(self, position_id: str, exit_price: float) -> TradeResult:
        """
        Close a position and calculate profit/loss.
        
        P&L Formula (from Property 5):
        profit_loss = (exit_price - entry_price) * size * direction - fees
        where:
            - direction = 1 for long, -1 for short
            - fees = entry_price * size * fee_rate + exit_price * size * fee_rate
        
        Args:
            position_id: ID of the position to close
            exit_price: Exit price
            
        Returns:
            TradeResult with P&L calculation
            
        Raises:
            ValueError: If position not found or exit_price is non-positive
        """
        if position_id not in self._positions:
            raise ValueError(f"Position not found: {position_id}")
        if exit_price <= 0:
            raise ValueError(f"Exit price must be positive, got {exit_price}")
        
        position = self._positions.pop(position_id)
        
        # Direction multiplier: 1 for long, -1 for short
        direction_multiplier = 1 if position.direction == "long" else -1
        
        # Calculate fees (entry + exit)
        entry_fee = position.entry_price * position.size * self.trading_fee
        exit_fee = exit_price * position.size * self.trading_fee
        total_fees = entry_fee + exit_fee
        
        # Calculate P&L: (X - E) * S * D - fees
        gross_pnl = (exit_price - position.entry_price) * position.size * direction_multiplier
        profit_loss = gross_pnl - total_fees
        
        # Calculate percentage P&L (relative to position value at entry)
        position_value = position.entry_price * position.size
        profit_loss_percent = (profit_loss / position_value) * 100 if position_value > 0 else 0.0
        
        # Calculate duration
        exit_time = datetime.now()
        duration = exit_time - position.entry_time
        
        return TradeResult(
            position_id=position_id,
            entry_price=position.entry_price,
            exit_price=exit_price,
            direction=position.direction,
            size=position.size,
            profit_loss=profit_loss,
            profit_loss_percent=profit_loss_percent,
            fees=total_fees,
            duration=duration
        )
    
    def get_unrealized_pnl(self, current_price: float) -> float:
        """
        Calculate unrealized P&L for all open positions.
        
        Args:
            current_price: Current market price
            
        Returns:
            Total unrealized P&L across all positions
        """
        if current_price <= 0:
            raise ValueError(f"Current price must be positive, got {current_price}")
        
        total_unrealized_pnl = 0.0
        
        for position in self._positions.values():
            direction_multiplier = 1 if position.direction == "long" else -1
            unrealized = (current_price - position.entry_price) * position.size * direction_multiplier
            total_unrealized_pnl += unrealized
        
        return total_unrealized_pnl
    
    def get_position_by_id(self, position_id: str) -> Optional[Position]:
        """
        Retrieve a position by its identifier.
        
        Args:
            position_id: The position ID to look up
            
        Returns:
            The Position if found, None otherwise
        """
        return self._positions.get(position_id)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """
        Get all open positions.
        
        Returns:
            Dictionary of position_id -> Position
        """
        return dict(self._positions)
    
    def has_open_positions(self) -> bool:
        """Check if there are any open positions."""
        return len(self._positions) > 0
    
    def clear_all_positions(self) -> None:
        """Clear all positions (useful for environment reset)."""
        self._positions.clear()
