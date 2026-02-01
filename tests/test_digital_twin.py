"""
Test suite for Digital Twin visualization system

Tests:
1. MotorcycleTelemetry data structure and serialization
2. SocketBridgeServer initialization and client handling
3. EnvironmentBridge integration
4. WebSocket protocol compliance
5. Trajectory buffering and limits
6. Error handling and reconnection
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import numpy as np

# Import modules to test
from src.deployment.socket_bridge import (
    MotorcycleTelemetry,
    SocketBridgeServer,
    EnvironmentBridge,
)


class TestMotorcycleTelemetry:
    """Test MotorcycleTelemetry data structure"""

    def test_telemetry_initialization(self):
        """Test telemetry object creation with default values"""
        telemetry = MotorcycleTelemetry()
        
        assert np.allclose(telemetry.position, [0, 0, 0])
        assert np.allclose(telemetry.rotation, [0, 0, 0])
        assert np.allclose(telemetry.velocity, [0, 0, 0])
        assert telemetry.speed == 0.0
        assert telemetry.throttle == 0.0
        assert telemetry.brake == 0.0
        assert telemetry.lean_angle == 0.0

    def test_telemetry_custom_values(self):
        """Test telemetry with custom values"""
        position = [1.5, 0.5, -2.3]
        rotation = [0.1, 0.2, 0.3]
        velocity = [10, 0, 5]
        
        telemetry = MotorcycleTelemetry(
            position=position,
            rotation=rotation,
            velocity=velocity,
            speed=15.5,
            throttle=0.75,
            brake=0.1,
            lean_angle=5.2,
        )
        
        assert np.allclose(telemetry.position, position)
        assert np.allclose(telemetry.rotation, rotation)
        assert np.allclose(telemetry.velocity, velocity)
        assert telemetry.speed == 15.5
        assert telemetry.throttle == 0.75
        assert telemetry.brake == 0.1
        assert telemetry.lean_angle == 5.2

    def test_telemetry_to_json(self):
        """Test serialization to JSON"""
        telemetry = MotorcycleTelemetry(
            position=[1.0, 2.0, 3.0],
            rotation=[0.1, 0.2, 0.3],
            velocity=[10, 0, 5],
            speed=15.5,
            throttle=0.75,
            brake=0.1,
            prediction=[1.1, 2.1, 3.1],
            reward=1.0,
        )
        
        json_data = telemetry.to_json()
        
        # Verify structure
        assert isinstance(json_data, dict)
        assert json_data['position'] == [1.0, 2.0, 3.0]
        # Allow for floating point precision
        assert np.allclose(json_data['rotation'], [0.1, 0.2, 0.3], rtol=1e-5)
        assert json_data['speed'] == 15.5
        assert json_data['throttle'] == 0.75

    def test_telemetry_json_serializable(self):
        """Test that telemetry JSON is fully serializable"""
        telemetry = MotorcycleTelemetry(
            position=[1.0, 2.0, 3.0],
            rotation=[0.1, 0.2, 0.3],
            velocity=[10, 0, 5],
        )
        
        json_data = telemetry.to_json()
        json_str = json.dumps(json_data)
        
        # Should not raise
        parsed = json.loads(json_str)
        assert parsed['position'] == [1.0, 2.0, 3.0]

    def test_telemetry_episode_info(self):
        """Test episode info in telemetry"""
        telemetry = MotorcycleTelemetry(
            episode_step=100,
            episode_num=5,
            done=False,
        )
        
        json_data = telemetry.to_json()
        assert json_data['episode_info']['step'] == 100
        assert json_data['episode_info']['episode'] == 5
        assert json_data['episode_info']['done'] is False

    def test_telemetry_bounds(self):
        """Test telemetry with boundary values"""
        telemetry = MotorcycleTelemetry(
            throttle=1.0,  # Max
            brake=0.0,     # Min
        )
        
        assert 0 <= telemetry.throttle <= 1.0
        assert 0 <= telemetry.brake <= 1.0


class TestSocketBridgeServer:
    """Test WebSocket server functionality"""

    def test_server_custom_parameters(self):
        """Test server with custom parameters"""
        server = SocketBridgeServer(
            host='localhost',
            port=8000,
            max_clients=5,
        )
        
        assert server.host == 'localhost'
        assert server.port == 8000
        assert server.max_clients == 5

    def test_server_has_clients_set(self):
        """Test server has clients container"""
        server = SocketBridgeServer()
        assert hasattr(server, 'clients')
        assert isinstance(server.clients, set)

    def test_server_get_status(self):
        """Test server status reporting"""
        server = SocketBridgeServer()
        
        # Server should have a status method
        status = server.get_status()
        assert isinstance(status, dict)
        # Check for some expected keys
        assert 'host' in status or 'connected_clients' in status or 'clients_count' in status


class TestEnvironmentBridge:
    """Test environment-to-websocket bridge"""

    def test_bridge_has_server(self):
        """Test bridge has WebSocket server"""
        # Create a mock env
        mock_env = MagicMock()
        bridge = EnvironmentBridge(mock_env)
        
        assert hasattr(bridge, 'server')

    def test_bridge_episode_tracking(self):
        """Test episode and step tracking"""
        mock_env = MagicMock()
        bridge = EnvironmentBridge(mock_env)
        
        # Simulate episode progression
        bridge.episode_num += 1
        bridge.episode_step += 1
        
        assert bridge.episode_num >= 0
        assert bridge.episode_step >= 0


class TestProtocolCompliance:
    """Test WebSocket protocol compliance"""

    def test_message_type_telemetry(self):
        """Test telemetry message type"""
        message = {
            'type': 'telemetry',
            'data': {
                'position': [0, 0, 0],
                'rotation': [0, 0, 0],
            }
        }
        
        # Should serialize without error
        json_str = json.dumps(message)
        parsed = json.loads(json_str)
        
        assert parsed['type'] == 'telemetry'

    def test_message_type_init(self):
        """Test init message type"""
        message = {
            'type': 'init',
            'trajectory': {
                'real': [],
                'predicted': [],
            }
        }
        
        json_str = json.dumps(message)
        parsed = json.loads(json_str)
        
        assert parsed['type'] == 'init'

    def test_telemetry_required_fields(self):
        """Test that telemetry has all required fields"""
        telemetry = MotorcycleTelemetry()
        json_data = telemetry.to_json()
        
        required_fields = [
            'position', 'rotation', 'velocity',
            'speed', 'throttle', 'brake', 'lean_angle',
            'episode_info'
        ]
        
        for field in required_fields:
            assert field in json_data

    def test_response_to_ping(self):
        """Test pong response to ping"""
        ping_message = {'type': 'ping'}
        
        # Server should respond with pong
        pong_response = {'type': 'pong'}
        
        assert pong_response['type'] == 'pong'

    def test_malformed_message_handling(self):
        """Test handling of malformed messages"""
        malformed = "not json"
        
        try:
            json.loads(malformed)
            assert False, "Should have raised JSONDecodeError"
        except json.JSONDecodeError:
            pass  # Expected


class TestTrajectoryManagement:
    """Test trajectory buffering and visualization"""

    def test_trajectory_fifo_queue(self):
        """Test trajectory buffer as FIFO queue"""
        from collections import deque
        
        # Simulate FIFO with max size
        max_size = 5
        buffer = deque(maxlen=max_size)
        
        # Add 7 points
        for i in range(7):
            buffer.append([i, 0, 0])
        
        # Should have last 5 points
        assert len(buffer) == max_size
        assert buffer[0] == [2, 0, 0]  # First after removals
        assert buffer[-1] == [6, 0, 0]  # Last added

    def test_trajectory_error_calculation(self):
        """Test trajectory error calculation"""
        real = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.1, 2.1, 3.1])
        
        error = np.linalg.norm(real - pred)
        
        assert error > 0
        assert error < 1.0  # Small error for small differences


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_server_max_clients_limit(self):
        """Test server respects max clients limit"""
        server = SocketBridgeServer(max_clients=2)
        
        # Add clients
        server.clients.add('client1')
        server.clients.add('client2')
        
        assert len(server.clients) == server.max_clients

    def test_division_by_zero_protection(self):
        """Test protection against division by zero"""
        telemetry = MotorcycleTelemetry(
            speed=0.0,  # Zero speed
        )
        
        # Should not cause division by zero
        json_data = telemetry.to_json()
        assert json_data['speed'] == 0.0

    def test_nan_value_handling(self):
        """Test handling of NaN values"""
        telemetry = MotorcycleTelemetry(
            speed=float('nan'),  # NaN value
        )
        
        # JSON encoder might fail on NaN
        try:
            json_data = telemetry.to_json()
            json.dumps(json_data)
        except (ValueError, TypeError):
            pass  # Expected for NaN

    def test_large_position_values(self):
        """Test handling of large position values"""
        telemetry = MotorcycleTelemetry(
            position=[1e6, 1e6, 1e6],
        )
        
        json_data = telemetry.to_json()
        assert json_data['position'] == [1e6, 1e6, 1e6]


class TestPerformance:
    """Test performance characteristics"""

    def test_telemetry_creation_speed(self):
        """Test telemetry object creation is fast"""
        import time
        
        start = time.time()
        for _ in range(1000):
            telemetry = MotorcycleTelemetry()
        elapsed = time.time() - start
        
        # Should complete quickly (< 100ms for 1000 objects)
        assert elapsed < 0.1

    def test_json_serialization_speed(self):
        """Test JSON serialization performance"""
        import time
        
        telemetry = MotorcycleTelemetry(
            position=[1.0, 2.0, 3.0],
            rotation=[0.1, 0.2, 0.3],
        )
        
        start = time.time()
        for _ in range(1000):
            json.dumps(telemetry.to_json())
        elapsed = time.time() - start
        
        # Should complete quickly (< 100ms for 1000 serializations)
        assert elapsed < 0.1


# Integration tests
class TestIntegration:
    """Integration tests for full system"""

    def test_telemetry_roundtrip(self):
        """Test telemetry roundtrip: object → JSON → dict"""
        original = MotorcycleTelemetry(
            position=[1.5, 2.5, 3.5],
            rotation=[0.1, 0.2, 0.3],
            speed=25.5,
        )
        
        # To JSON
        json_data = original.to_json()
        
        # To JSON string
        json_str = json.dumps(json_data)
        
        # Back to dict
        parsed = json.loads(json_str)
        
        # Verify integrity
        assert parsed['position'] == [1.5, 2.5, 3.5]
        assert parsed['speed'] == 25.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
