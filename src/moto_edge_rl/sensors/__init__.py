"""Sensor data processing and fusion.

This module handles data from various sensors including IMU, GPS, and
telemetry systems.
"""

from typing import Dict, Any, Optional
import numpy as np
from dataclasses import dataclass

__all__ = ["SensorData", "IMUSensor", "GPSSensor", "SensorFusion"]


@dataclass
class SensorData:
    """Container for sensor readings."""
    
    timestamp: float
    data: Dict[str, Any]
    sensor_type: str


class IMUSensor:
    """Inertial Measurement Unit sensor interface."""
    
    def __init__(self, sampling_rate: int = 100):
        """Initialize IMU sensor.
        
        Args:
            sampling_rate: Sampling rate in Hz.
        """
        self.sampling_rate = sampling_rate
        self.is_active = False
    
    def read(self) -> Optional[SensorData]:
        """Read current IMU data.
        
        Returns:
            SensorData object with IMU readings, or None if not active.
        """
        if not self.is_active:
            return None
        
        # Placeholder data
        data = {
            "acceleration": np.zeros(3),
            "gyroscope": np.zeros(3),
            "magnetometer": np.zeros(3)
        }
        
        return SensorData(
            timestamp=0.0,
            data=data,
            sensor_type="IMU"
        )


class GPSSensor:
    """GPS sensor interface."""
    
    def __init__(self, sampling_rate: int = 10):
        """Initialize GPS sensor.
        
        Args:
            sampling_rate: Sampling rate in Hz.
        """
        self.sampling_rate = sampling_rate
        self.is_active = False
    
    def read(self) -> Optional[SensorData]:
        """Read current GPS data.
        
        Returns:
            SensorData object with GPS readings, or None if not active.
        """
        if not self.is_active:
            return None
        
        # Placeholder data
        data = {
            "latitude": 0.0,
            "longitude": 0.0,
            "altitude": 0.0,
            "speed": 0.0
        }
        
        return SensorData(
            timestamp=0.0,
            data=data,
            sensor_type="GPS"
        )


class SensorFusion:
    """Fuse data from multiple sensors."""
    
    def __init__(self):
        """Initialize sensor fusion system."""
        self.sensors = {}
    
    def add_sensor(self, name: str, sensor: Any) -> None:
        """Add a sensor to the fusion system.
        
        Args:
            name: Name identifier for the sensor.
            sensor: The sensor object.
        """
        self.sensors[name] = sensor
    
    def get_fused_data(self) -> Dict[str, Any]:
        """Get fused data from all sensors.
        
        Returns:
            Dictionary containing fused sensor data.
        """
        fused_data = {}
        
        for name, sensor in self.sensors.items():
            data = sensor.read()
            if data is not None:
                fused_data[name] = data
        
        return fused_data
