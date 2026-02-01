"""Haptic feedback controllers and patterns.

This module handles the translation of RL agent actions into haptic feedback
signals for coaching the rider.
"""

from typing import Dict, Any
from enum import Enum

__all__ = ["HapticController", "FeedbackPattern", "HapticSignal"]


class FeedbackPattern(Enum):
    """Enum for different haptic feedback patterns."""
    
    GENTLE_VIBRATION = "gentle"
    STRONG_VIBRATION = "strong"
    PULSE = "pulse"
    CONTINUOUS = "continuous"
    WARNING = "warning"


class HapticSignal:
    """Representation of a haptic signal."""
    
    def __init__(
        self,
        pattern: FeedbackPattern,
        intensity: float,
        duration: float
    ):
        """Initialize haptic signal.
        
        Args:
            pattern: Type of feedback pattern.
            intensity: Signal intensity (0.0 to 1.0).
            duration: Duration in seconds.
        """
        self.pattern = pattern
        self.intensity = max(0.0, min(1.0, intensity))
        self.duration = duration
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary format.
        
        Returns:
            Dictionary representation of the signal.
        """
        return {
            "pattern": self.pattern.value,
            "intensity": self.intensity,
            "duration": self.duration
        }


class HapticController:
    """Controller for managing haptic feedback devices."""
    
    def __init__(self, device_id: str = "default"):
        """Initialize haptic controller.
        
        Args:
            device_id: ID of the haptic device to control.
        """
        self.device_id = device_id
        self.is_connected = False
    
    def connect(self) -> bool:
        """Connect to the haptic device.
        
        Returns:
            True if connection successful, False otherwise.
        """
        # Placeholder for actual device connection
        self.is_connected = True
        return self.is_connected
    
    def send_signal(self, signal: HapticSignal) -> bool:
        """Send a haptic signal to the device.
        
        Args:
            signal: The haptic signal to send.
            
        Returns:
            True if signal sent successfully, False otherwise.
        """
        if not self.is_connected:
            return False
        
        # Placeholder for actual signal transmission
        return True
    
    def disconnect(self) -> None:
        """Disconnect from the haptic device."""
        self.is_connected = False
