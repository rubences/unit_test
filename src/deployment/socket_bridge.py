"""
Socket Bridge: Real-time telemetry streaming from Gymnasium environment to 3D visualizer.

Architecture:
- Server: WebSocket server that streams motorcycle state (position, rotation, velocity)
- Protocol: JSON over WebSocket
- Client: HTML/JS (Three.js) or Unity C#

Data Format (sent every env.step()):
{
    "timestamp": 1234567890.123,
    "position": [x, y, z],           # meters
    "rotation": [roll, pitch, yaw],  # radians (Euler angles)
    "velocity": [vx, vy, vz],        # m/s
    "speed": scalar,                 # current speed
    "throttle": [0, 1],              # throttle position
    "brake": [0, 1],                 # brake pressure
    "lean_angle": scalar,            # lean angle (degrees)
    "track_coords": [distance_along, lateral_offset],  # for trajectory
    "prediction": [x_pred, y_pred, z_pred],  # AI predicted position
    "reward": scalar,                # Episode reward
    "episode_info": {
        "step": int,
        "episode": int,
        "done": bool,
    }
}
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    from websockets.exceptions import ConnectionClosed
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    WebSocketServerProtocol = Any  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class MotorcycleTelemetry:
    """Telemetry data container for motorcycle state."""

    def __init__(
        self,
        position: Tuple[float, float, float] = (0, 0, 0),
        rotation: Tuple[float, float, float] = (0, 0, 0),
        velocity: Tuple[float, float, float] = (0, 0, 0),
        speed: float = 0.0,
        throttle: float = 0.0,
        brake: float = 0.0,
        lean_angle: float = 0.0,
        track_coords: Tuple[float, float] = (0, 0),
        prediction: Tuple[float, float, float] = (0, 0, 0),
        reward: float = 0.0,
        episode_step: int = 0,
        episode_num: int = 0,
        done: bool = False,
    ):
        """Initialize telemetry data.

        Args:
            position: (x, y, z) in meters
            rotation: (roll, pitch, yaw) in radians
            velocity: (vx, vy, vz) in m/s
            speed: scalar speed in m/s
            throttle: throttle [0, 1]
            brake: brake [0, 1]
            lean_angle: lean angle in degrees
            track_coords: (distance_along_track, lateral_offset)
            prediction: (x_pred, y_pred, z_pred) AI predicted position
            reward: episode reward
            episode_step: step count in episode
            episode_num: episode number
            done: episode done flag
        """
        self.timestamp = time.time()
        self.position = np.array(position, dtype=np.float32)
        self.rotation = np.array(rotation, dtype=np.float32)
        self.velocity = np.array(velocity, dtype=np.float32)
        self.speed = float(speed)
        self.throttle = float(throttle)
        self.brake = float(brake)
        self.lean_angle = float(lean_angle)
        self.track_coords = np.array(track_coords, dtype=np.float32)
        self.prediction = np.array(prediction, dtype=np.float32)
        self.reward = float(reward)
        self.episode_step = int(episode_step)
        self.episode_num = int(episode_num)
        self.done = bool(done)

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary.

        Returns:
            Dictionary ready for JSON serialization
        """
        return {
            "timestamp": self.timestamp,
            "position": self.position.tolist(),
            "rotation": self.rotation.tolist(),
            "velocity": self.velocity.tolist(),
            "speed": self.speed,
            "throttle": self.throttle,
            "brake": self.brake,
            "lean_angle": self.lean_angle,
            "track_coords": self.track_coords.tolist(),
            "prediction": self.prediction.tolist(),
            "reward": self.reward,
            "episode_info": {
                "step": self.episode_step,
                "episode": self.episode_num,
                "done": self.done,
            },
        }


class SocketBridgeServer:
    """WebSocket server for streaming motorcycle telemetry.

    Usage:
        server = SocketBridgeServer(host="localhost", port=5555)
        asyncio.run(server.start())

        # In env.step() loop:
        telemetry = MotorcycleTelemetry(position=(1,2,3), ...)
        server.broadcast(telemetry)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        max_clients: int = 10,
        buffer_size: int = 1000,
    ):
        """Initialize WebSocket server.

        Args:
            host: Server host
            port: Server port
            max_clients: Maximum concurrent clients
            buffer_size: Trajectory buffer size
        """
        self.host = host
        self.port = port
        self.max_clients = max_clients
        self.buffer_size = buffer_size

        self.clients: set[WebSocketServerProtocol] = set()
        self.trajectory_buffer = {
            "real": [],       # Real trajectory (green)
            "predicted": [],  # Predicted trajectory (red)
        }
        self.current_telemetry: Optional[MotorcycleTelemetry] = None
        self.message_count = 0
        self.start_time = time.time()

        if not WEBSOCKETS_AVAILABLE:
            logger.error("websockets library not available. Install: pip install websockets")

    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new client connection.

        Args:
            websocket: Client WebSocket connection
            path: Connection path
        """
        try:
            logger.info(f"Client connected from {websocket.remote_address}")
            self.clients.add(websocket)

            # Send initial state (trajectory history)
            init_message = {
                "type": "init",
                "trajectory": {
                    "real": self.trajectory_buffer["real"][-100:],  # Last 100 points
                    "predicted": self.trajectory_buffer["predicted"][-100:],
                },
            }
            await websocket.send(json.dumps(init_message))

            # Keep connection alive
            try:
                async for message in websocket:
                    # Clients can send requests (e.g., "reset", "pause")
                    await self.handle_client_message(message, websocket)
            except Exception:  # ConnectionClosed
                pass

        except Exception as e:
            logger.error(f"Error handling client: {e}")

        finally:
            self.clients.discard(websocket)
            logger.info(f"Client disconnected: {websocket.remote_address}")

    async def handle_client_message(self, message: str, websocket: WebSocketServerProtocol):
        """Handle incoming client message.

        Args:
            message: JSON message from client
            websocket: Client WebSocket
        """
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")

            if msg_type == "request_state":
                # Client requesting current state
                if self.current_telemetry:
                    response = {
                        "type": "telemetry",
                        "data": self.current_telemetry.to_json(),
                    }
                    await websocket.send(json.dumps(response))

            elif msg_type == "ping":
                # Keep-alive ping
                await websocket.send(json.dumps({"type": "pong"}))

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from client: {message}")

    async def broadcast(self, telemetry: MotorcycleTelemetry):
        """Broadcast telemetry to all connected clients.

        Args:
            telemetry: Motorcycle telemetry data
        """
        self.current_telemetry = telemetry
        self.message_count += 1

        # Update trajectory buffers
        pos = telemetry.position
        pred = telemetry.prediction

        self.trajectory_buffer["real"].append(pos.tolist())
        self.trajectory_buffer["predicted"].append(pred.tolist())

        # Keep buffers manageable
        if len(self.trajectory_buffer["real"]) > self.buffer_size:
            self.trajectory_buffer["real"].pop(0)
            self.trajectory_buffer["predicted"].pop(0)

        # Prepare message
        message = {
            "type": "telemetry",
            "data": telemetry.to_json(),
            "trajectory_length": {
                "real": len(self.trajectory_buffer["real"]),
                "predicted": len(self.trajectory_buffer["predicted"]),
            },
        }

        # Send to all connected clients
        if self.clients:
            message_json = json.dumps(message)
            disconnected = set()

            for client in self.clients:
                try:
                    await client.send(message_json)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(client)
                except Exception as e:
                    logger.error(f"Error sending to client: {e}")
                    disconnected.add(client)

            # Clean up disconnected clients
            self.clients -= disconnected

            if self.message_count % 100 == 0:
                elapsed = time.time() - self.start_time
                rate = self.message_count / elapsed
                logger.info(f"Broadcast: {self.message_count} msgs, "
                          f"{len(self.clients)} clients, {rate:.1f} Hz")

    async def start(self):
        """Start WebSocket server."""
        if not WEBSOCKETS_AVAILABLE:
            logger.error("Cannot start server: websockets not installed")
            return

        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")

        async with websockets.serve(self.handle_client, self.host, self.port):
            logger.info(f"✓ Server listening on ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever

    def get_status(self) -> Dict[str, Any]:
        """Get server status.

        Returns:
            Status dictionary
        """
        return {
            "host": self.host,
            "port": self.port,
            "connected_clients": len(self.clients),
            "messages_sent": self.message_count,
            "trajectory_size": {
                "real": len(self.trajectory_buffer["real"]),
                "predicted": len(self.trajectory_buffer["predicted"]),
            },
            "uptime_seconds": time.time() - self.start_time,
        }


class EnvironmentBridge:
    """Bridge between Gymnasium environment and WebSocket server.

    Usage:
        bridge = EnvironmentBridge(env, port=5555)
        
        obs, info = env.reset()
        for _ in range(1000):
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Extract telemetry and broadcast
            bridge.update(obs, action, reward, terminated, truncated, info)
            
            if terminated or truncated:
                obs, info = env.reset()
    """

    def __init__(
        self,
        env,
        port: int = 5555,
        host: str = "localhost",
        obs_indices: Optional[Dict[str, int]] = None,
    ):
        """Initialize environment bridge.

        Args:
            env: Gymnasium environment
            port: WebSocket server port
            host: WebSocket server host
            obs_indices: Mapping observation indices to telemetry fields
                        Default: assumes obs = [x, y, z, roll, pitch, yaw, ...]
        """
        self.env = env
        self.port = port
        self.host = host
        self.server = SocketBridgeServer(host=host, port=port)

        # Default observation indices
        self.obs_indices = obs_indices or {
            "x": 0,
            "y": 1,
            "z": 2,
            "roll": 3,
            "pitch": 4,
            "yaw": 5,
            "vx": 6,
            "vy": 7,
            "vz": 8,
        }

        self.episode_num = 0
        self.episode_step = 0
        self.prediction_buffer = np.zeros(3)  # For AI predictions

        logger.info(f"EnvironmentBridge initialized on {host}:{port}")

    def extract_telemetry(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict,
    ) -> MotorcycleTelemetry:
        """Extract telemetry from environment observation and step results.

        Args:
            obs: Observation from env
            action: Action taken
            reward: Reward received
            terminated: Episode termination flag
            truncated: Episode truncation flag
            info: Info dictionary from env

        Returns:
            MotorcycleTelemetry object
        """
        # Extract position
        obs_array = np.array(obs) if not isinstance(obs, np.ndarray) else obs
        
        position = tuple(obs_array[i] for i in [
            self.obs_indices.get("x", 0),
            self.obs_indices.get("y", 1),
            self.obs_indices.get("z", 2),
        ]) if len(obs_array) > 2 else (0, 0, 0)

        # Extract rotation
        rotation = tuple(obs_array[i] for i in [
            self.obs_indices.get("roll", 3),
            self.obs_indices.get("pitch", 4),
            self.obs_indices.get("yaw", 5),
        ]) if len(obs_array) > 5 else (0, 0, 0)

        # Extract velocity
        velocity = tuple(obs_array[i] for i in [
            self.obs_indices.get("vx", 6),
            self.obs_indices.get("vy", 7),
            self.obs_indices.get("vz", 8),
        ]) if len(obs_array) > 8 else (0, 0, 0)

        speed = float(np.linalg.norm(velocity)) if velocity else 0.0

        # Extract control inputs
        action_array = np.array(action) if not isinstance(action, np.ndarray) else action
        throttle = float(action_array[0]) if len(action_array) > 0 else 0.0
        brake = float(action_array[1]) if len(action_array) > 1 else 0.0

        # Track coordinates (simplified)
        track_coords = (float(position[0]) if position else 0.0, 
                       float(position[1]) if position else 0.0)

        # AI prediction (can be overridden with actual model prediction)
        prediction = tuple(self.prediction_buffer)

        done = terminated or truncated

        return MotorcycleTelemetry(
            position=position,
            rotation=rotation,
            velocity=velocity,
            speed=speed,
            throttle=throttle,
            brake=brake,
            lean_angle=float(rotation[0]) * 180 / np.pi if rotation else 0.0,
            track_coords=track_coords,
            prediction=prediction,
            reward=reward,
            episode_step=self.episode_step,
            episode_num=self.episode_num,
            done=done,
        )

    async def update(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict,
    ):
        """Update telemetry and broadcast to clients.

        Args:
            obs: Observation from env
            action: Action taken
            reward: Reward received
            terminated: Episode termination flag
            truncated: Episode truncation flag
            info: Info dictionary from env
        """
        telemetry = self.extract_telemetry(obs, action, reward, terminated, truncated, info)
        await self.server.broadcast(telemetry)

        self.episode_step += 1

        if terminated or truncated:
            self.episode_num += 1
            self.episode_step = 0

    def set_prediction(self, prediction: Tuple[float, float, float]):
        """Set AI prediction for current step.

        Args:
            prediction: (x_pred, y_pred, z_pred)
        """
        self.prediction_buffer = np.array(prediction, dtype=np.float32)

    async def start(self):
        """Start the bridge server."""
        await self.server.start()

    def get_status(self) -> Dict[str, Any]:
        """Get bridge status.

        Returns:
            Status dictionary
        """
        return {
            "server": self.server.get_status(),
            "episode": self.episode_num,
            "episode_step": self.episode_step,
        }


# Example usage
async def example_demo():
    """Demonstrate socket bridge usage."""
    server = SocketBridgeServer(host="localhost", port=5555)

    # Simulate telemetry stream
    async def simulate_telemetry():
        """Simulate motorcycle movement."""
        for step in range(1000):
            # Simulate circular motion
            angle = step * 0.01
            x = 10 * np.cos(angle)
            y = 10 * np.sin(angle)
            z = 0.5

            # Simulated prediction (offset from real)
            x_pred = x + 0.5 * np.sin(step * 0.05)
            y_pred = y + 0.5 * np.cos(step * 0.05)

            telemetry = MotorcycleTelemetry(
                position=(x, y, z),
                rotation=(0.1 * np.sin(angle), 0.05, angle),
                velocity=(np.cos(angle), np.sin(angle), 0),
                speed=1.0,
                throttle=0.7,
                brake=0.0,
                lean_angle=5.0 * np.sin(angle),
                prediction=(x_pred, y_pred, z),
                reward=1.0,
                episode_step=step,
                episode_num=0,
                done=False,
            )

            await server.broadcast(telemetry)
            await asyncio.sleep(0.01)  # 100 Hz

    # Run server and simulation
    server_task = asyncio.create_task(server.start())
    simulation_task = asyncio.create_task(simulate_telemetry())

    await asyncio.gather(server_task, simulation_task)


if __name__ == "__main__":
    if WEBSOCKETS_AVAILABLE:
        asyncio.run(example_demo())
    else:
        print("ERROR: websockets not installed. Run: pip install websockets")
