"""Tests for haptic feedback controllers."""

import pytest
from moto_edge_rl.haptic import (
    HapticController,
    HapticSignal,
    FeedbackPattern
)


class TestFeedbackPattern:
    """Tests for FeedbackPattern enum."""
    
    def test_feedback_patterns_exist(self):
        """Test that all feedback patterns are defined."""
        assert hasattr(FeedbackPattern, "GENTLE_VIBRATION")
        assert hasattr(FeedbackPattern, "STRONG_VIBRATION")
        assert hasattr(FeedbackPattern, "PULSE")
        assert hasattr(FeedbackPattern, "CONTINUOUS")
        assert hasattr(FeedbackPattern, "WARNING")


class TestHapticSignal:
    """Tests for HapticSignal class."""
    
    def test_init(self):
        """Test haptic signal initialization."""
        signal = HapticSignal(
            pattern=FeedbackPattern.GENTLE_VIBRATION,
            intensity=0.5,
            duration=1.0
        )
        assert signal.pattern == FeedbackPattern.GENTLE_VIBRATION
        assert signal.intensity == 0.5
        assert signal.duration == 1.0
    
    def test_intensity_clamping_upper(self):
        """Test that intensity is clamped to 1.0."""
        signal = HapticSignal(
            pattern=FeedbackPattern.PULSE,
            intensity=1.5,
            duration=0.5
        )
        assert signal.intensity == 1.0
    
    def test_intensity_clamping_lower(self):
        """Test that intensity is clamped to 0.0."""
        signal = HapticSignal(
            pattern=FeedbackPattern.PULSE,
            intensity=-0.5,
            duration=0.5
        )
        assert signal.intensity == 0.0
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        signal = HapticSignal(
            pattern=FeedbackPattern.WARNING,
            intensity=0.8,
            duration=0.2
        )
        signal_dict = signal.to_dict()
        
        assert signal_dict["pattern"] == "warning"
        assert signal_dict["intensity"] == 0.8
        assert signal_dict["duration"] == 0.2


class TestHapticController:
    """Tests for HapticController class."""
    
    def test_init(self):
        """Test controller initialization."""
        controller = HapticController(device_id="test_device")
        assert controller.device_id == "test_device"
        assert controller.is_connected is False
    
    def test_connect(self):
        """Test device connection."""
        controller = HapticController()
        result = controller.connect()
        assert result is True
        assert controller.is_connected is True
    
    def test_send_signal_when_connected(self):
        """Test sending signal when connected."""
        controller = HapticController()
        controller.connect()
        
        signal = HapticSignal(
            pattern=FeedbackPattern.PULSE,
            intensity=0.5,
            duration=1.0
        )
        result = controller.send_signal(signal)
        assert result is True
    
    def test_send_signal_when_disconnected(self):
        """Test sending signal when not connected."""
        controller = HapticController()
        signal = HapticSignal(
            pattern=FeedbackPattern.PULSE,
            intensity=0.5,
            duration=1.0
        )
        result = controller.send_signal(signal)
        assert result is False
    
    def test_disconnect(self):
        """Test device disconnection."""
        controller = HapticController()
        controller.connect()
        controller.disconnect()
        assert controller.is_connected is False
