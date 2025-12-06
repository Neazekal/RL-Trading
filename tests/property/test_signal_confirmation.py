"""Property-based tests for Signal Confirmation Layer.

**Feature: rl-crypto-trading-agents, Property 20: Signal Confirmation Execution**
**Feature: rl-crypto-trading-agents, Property 21: Signal Confirmation Hold Before N**
**Feature: rl-crypto-trading-agents, Property 22: Signal Confirmation Reset on Interruption**
"""

from hypothesis import given, strategies as st, settings
import pytest

from src.agents.signal_confirmation import SignalConfirmationLayer


# Strategy for generating valid required_confirmations values
required_confirmations_strategy = st.integers(min_value=1, max_value=10)

# Strategy for generating open signals (1=open_long, 2=open_short)
open_signal_strategy = st.sampled_from([1, 2])

# Strategy for generating any valid action (0=hold, 1=open_long, 2=open_short)
action_strategy = st.sampled_from([0, 1, 2])


class TestSignalConfirmationExecution:
    """Test Property 20: Signal Confirmation Execution.
    
    **Property 20: Signal Confirmation Execution**
    
    *For any* sequence of N consecutive identical open signals (all open_long 
    or all open_short), the Signal Confirmation Layer SHALL output the actual 
    open action only after the Nth signal.
    
    **Validates: Requirements 3.7, 3.8**
    """
    
    @given(
        required_confirmations=required_confirmations_strategy,
        signal=open_signal_strategy
    )
    @settings(max_examples=100)
    def test_executes_after_n_consecutive_signals(self, required_confirmations, signal):
        """Test that open action is executed only after N consecutive identical signals."""
        layer = SignalConfirmationLayer(required_confirmations)
        
        results = []
        for i in range(required_confirmations):
            result = layer.process_signal(signal)
            results.append(result)
        
        # All outputs before the Nth should be hold (0)
        for i in range(required_confirmations - 1):
            assert results[i] == 0, (
                f"Expected hold (0) at position {i}, got {results[i]}. "
                f"N={required_confirmations}, signal={signal}"
            )
        
        # The Nth output should be the actual signal
        assert results[-1] == signal, (
            f"Expected signal {signal} at position {required_confirmations - 1}, "
            f"got {results[-1]}. N={required_confirmations}"
        )
    
    @given(
        required_confirmations=required_confirmations_strategy,
        signal=open_signal_strategy,
        extra_signals=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=100)
    def test_resets_after_execution(self, required_confirmations, signal, extra_signals):
        """Test that counter resets after successful execution."""
        layer = SignalConfirmationLayer(required_confirmations)
        
        # First, reach N consecutive signals to trigger execution
        for _ in range(required_confirmations):
            layer.process_signal(signal)
        
        # After execution, the counter should be reset
        # So sending more signals should start counting from 1 again
        results = []
        for _ in range(extra_signals):
            result = layer.process_signal(signal)
            results.append(result)
        
        # If extra_signals < required_confirmations, all should be hold
        # If extra_signals >= required_confirmations, the Nth should execute
        for i, result in enumerate(results):
            position = i + 1  # 1-indexed position in new sequence
            if position < required_confirmations:
                assert result == 0, (
                    f"Expected hold (0) at new sequence position {position}, "
                    f"got {result}. N={required_confirmations}"
                )
            elif position == required_confirmations:
                assert result == signal, (
                    f"Expected signal {signal} at new sequence position {position}, "
                    f"got {result}. N={required_confirmations}"
                )


class TestSignalConfirmationHoldBeforeN:
    """Test Property 21: Signal Confirmation Hold Before N.
    
    **Property 21: Signal Confirmation Hold Before N**
    
    *For any* sequence of fewer than N consecutive identical open signals, 
    the Signal Confirmation Layer SHALL output hold action and increment 
    the counter.
    
    **Validates: Requirements 3.9**
    """
    
    @given(
        required_confirmations=st.integers(min_value=2, max_value=10),
        signal=open_signal_strategy
    )
    @settings(max_examples=100)
    def test_outputs_hold_before_n_signals(self, required_confirmations, signal):
        """Test that all outputs are hold before reaching N signals."""
        layer = SignalConfirmationLayer(required_confirmations)
        
        # Send fewer than N signals
        num_signals = required_confirmations - 1
        
        for i in range(num_signals):
            result = layer.process_signal(signal)
            assert result == 0, (
                f"Expected hold (0) at position {i}, got {result}. "
                f"N={required_confirmations}, signals_sent={i + 1}"
            )
    
    @given(
        required_confirmations=st.integers(min_value=2, max_value=10),
        signal=open_signal_strategy,
        num_signals=st.integers(min_value=1, max_value=9)
    )
    @settings(max_examples=100)
    def test_counter_increments_correctly(self, required_confirmations, signal, num_signals):
        """Test that counter increments with each consecutive signal."""
        # Ensure num_signals is less than required_confirmations
        if num_signals >= required_confirmations:
            num_signals = required_confirmations - 1
        
        layer = SignalConfirmationLayer(required_confirmations)
        
        for i in range(num_signals):
            layer.process_signal(signal)
            current_signal, count, required = layer.get_confirmation_progress()
            
            assert current_signal == signal, (
                f"Expected current_signal={signal}, got {current_signal}"
            )
            assert count == i + 1, (
                f"Expected count={i + 1}, got {count}"
            )
            assert required == required_confirmations, (
                f"Expected required={required_confirmations}, got {required}"
            )


class TestSignalConfirmationResetOnInterruption:
    """Test Property 22: Signal Confirmation Reset on Interruption.
    
    **Property 22: Signal Confirmation Reset on Interruption**
    
    *For any* sequence of consecutive signals that is interrupted by a 
    different action (including hold), the Signal Confirmation Layer 
    SHALL reset the counter to zero.
    
    **Validates: Requirements 3.10**
    """
    
    @given(
        required_confirmations=st.integers(min_value=2, max_value=10),
        signal=open_signal_strategy,
        num_signals_before=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=100)
    def test_hold_resets_counter(self, required_confirmations, signal, num_signals_before):
        """Test that hold action resets the signal counter."""
        # Ensure we don't reach N before interruption
        if num_signals_before >= required_confirmations:
            num_signals_before = required_confirmations - 1
        
        layer = SignalConfirmationLayer(required_confirmations)
        
        # Send some signals
        for _ in range(num_signals_before):
            layer.process_signal(signal)
        
        # Verify counter is at expected value
        _, count_before, _ = layer.get_confirmation_progress()
        assert count_before == num_signals_before
        
        # Send hold action
        result = layer.process_signal(0)  # hold
        assert result == 0
        
        # Verify counter is reset
        current_signal, count_after, _ = layer.get_confirmation_progress()
        assert current_signal is None, (
            f"Expected current_signal=None after hold, got {current_signal}"
        )
        assert count_after == 0, (
            f"Expected count=0 after hold, got {count_after}"
        )
    
    @given(
        required_confirmations=st.integers(min_value=2, max_value=10),
        num_signals_before=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=100)
    def test_different_signal_resets_and_starts_new_tracking(
        self, required_confirmations, num_signals_before
    ):
        """Test that different open signal resets counter and starts new tracking."""
        # Ensure we don't reach N before interruption
        if num_signals_before >= required_confirmations:
            num_signals_before = required_confirmations - 1
        
        layer = SignalConfirmationLayer(required_confirmations)
        
        # Start with open_long signals
        first_signal = 1  # open_long
        for _ in range(num_signals_before):
            layer.process_signal(first_signal)
        
        # Verify counter is at expected value
        current_before, count_before, _ = layer.get_confirmation_progress()
        assert current_before == first_signal
        assert count_before == num_signals_before
        
        # Send different signal (open_short)
        different_signal = 2  # open_short
        result = layer.process_signal(different_signal)
        
        # Should output hold since we're starting fresh
        assert result == 0
        
        # Verify counter is reset and tracking new signal
        current_after, count_after, _ = layer.get_confirmation_progress()
        assert current_after == different_signal, (
            f"Expected current_signal={different_signal}, got {current_after}"
        )
        assert count_after == 1, (
            f"Expected count=1 for new signal, got {count_after}"
        )
    
    @given(
        required_confirmations=st.integers(min_value=2, max_value=10),
        signal=open_signal_strategy,
        interruption_position=st.integers(min_value=1, max_value=8)
    )
    @settings(max_examples=100)
    def test_interrupted_sequence_never_executes(
        self, required_confirmations, signal, interruption_position
    ):
        """Test that interrupted sequence never triggers execution."""
        # Ensure interruption happens before N
        if interruption_position >= required_confirmations:
            interruption_position = required_confirmations - 1
        
        layer = SignalConfirmationLayer(required_confirmations)
        
        results = []
        
        # Send signals up to interruption point
        for i in range(interruption_position):
            result = layer.process_signal(signal)
            results.append(result)
        
        # Interrupt with hold
        result = layer.process_signal(0)
        results.append(result)
        
        # Continue with same signal (but counter was reset)
        remaining = required_confirmations - 1  # Not enough to trigger
        for _ in range(remaining):
            result = layer.process_signal(signal)
            results.append(result)
        
        # All results should be hold (0) since we never reached N consecutive
        for i, result in enumerate(results):
            assert result == 0, (
                f"Expected hold (0) at position {i}, got {result}. "
                f"Sequence was interrupted at position {interruption_position}"
            )
