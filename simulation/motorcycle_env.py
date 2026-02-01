"""
Custom Gymnasium Environment for Motorcycle Racing

This module implements a physics-based Gymnasium environment for training
reinforcement learning agents for motorcycle racing with haptic coaching.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional


class MotorcycleEnv(gym.Env):
    """
    Custom Gymnasium environment for motorcycle racing dynamics.
    
    This environment simulates motorcycle racing physics including:
    - Non-linear dynamics (Kamm Circle for tire friction)
    - Weight transfer during braking
    - Lean angle and lateral G-forces
    - Racing line optimization
    
    Observation Space:
        - velocity (m/s): Current speed [0, 100]
        - roll_angle (degrees): Lean angle [-60, 60]
        - lateral_g (G): Lateral acceleration [-2.0, 2.0]
        - distance_to_apex (m): Distance to next corner apex [0, 500]
        - throttle_position (0-1): Current throttle [0, 1]
        - brake_pressure (0-1): Current brake pressure [0, 1]
        - racing_line_deviation (m): Distance from optimal line [-10, 10]
        - tire_friction_usage (0-1): Percentage of grip used [0, 1]
    
    Action Space:
        - haptic_left_intensity (0-1): Left glove vibration intensity
        - haptic_right_intensity (0-1): Right glove vibration intensity
        - haptic_frequency (Hz): Vibration frequency [50, 300]
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode: Optional[str] = None, track_name: str = "silverstone"):
        """
        Initialize the motorcycle racing environment.
        
        Args:
            render_mode: Visualization mode ("human" or "rgb_array")
            track_name: Name of the racing track
        """
        super().__init__()
        
        self.track_name = track_name
        self.render_mode = render_mode
        
        # Physical constants and vehicle properties
        self.MAX_VELOCITY = 100.0  # m/s (~360 km/h)
        self.MAX_LEAN_ANGLE = 60.0  # degrees
        self.MAX_LATERAL_G = 2.5    # G-forces
        self.GRAVITY = 9.81  # m/s^2

        # 6-DOF simplified motorcycle parameters
        self.BASE_VEHICLE_MASS = 180.0  # kg (bike)
        self.BASE_RIDER_MASS = 75.0     # kg (pilot)
        self.WHEELBASE = 1.4            # m
        self.CG_HEIGHT = 0.55           # m
        self.LF = self.WHEELBASE * 0.5  # distance CG -> front
        self.LR = self.WHEELBASE * 0.5  # distance CG -> rear
        self.INERTIA_YAW = 120.0        # kg·m² (approx)
        self.INERTIA_ROLL = 85.0        # kg·m² (approx)

        # Pacejka Magic Formula parameters (moderate sport tire)
        self.Bx = 10.0
        self.Cx = 1.9
        self.Ex = 0.97
        self.By = 8.5
        self.Cy = 1.7
        self.Ey = 1.0
        self.MU_BASE = 1.2  # baseline asphalt friction

        # Domain randomization ranges
        self.MU_JITTER = 0.15     # ±15% friction variation
        self.RIDER_MASS_JITTER = 7.5  # ±7.5 kg variation
        
        # Define observation space (8-dimensional continuous state)
        self.observation_space = spaces.Box(
            low=np.array([
                0.0,      # velocity
                -60.0,    # roll_angle
                -2.0,     # lateral_g
                0.0,      # distance_to_apex
                0.0,      # throttle_position
                0.0,      # brake_pressure
                -10.0,    # racing_line_deviation
                0.0       # tire_friction_usage
            ], dtype=np.float32),
            high=np.array([
                100.0,    # velocity
                60.0,     # roll_angle
                2.0,      # lateral_g
                500.0,    # distance_to_apex
                1.0,      # throttle_position
                1.0,      # brake_pressure
                10.0,     # racing_line_deviation
                1.0       # tire_friction_usage
            ], dtype=np.float32),
            shape=(8,),
            dtype=np.float32
        )
        
        # Define action space (3-dimensional continuous actions for haptic feedback)
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 50.0], dtype=np.float32),   # [left, right, freq]
            high=np.array([1.0, 1.0, 300.0], dtype=np.float32),  # [left, right, freq]
            shape=(3,),
            dtype=np.float32
        )
        
        # Internal state
        self.state = None
        self.timestep = 0
        self.max_timesteps = 1000

        # Dynamic states for 6-DOF simplified model
        self.vx = 0.0  # longitudinal velocity (m/s)
        self.vy = 0.0  # lateral velocity (m/s)
        self.yaw_rate = 0.0  # yaw rate (rad/s)
        self.roll_angle_rad = 0.0  # roll angle (rad)
        self.roll_rate = 0.0  # roll rate (rad/s)
        self.total_mass = self.BASE_VEHICLE_MASS + self.BASE_RIDER_MASS
        self.surface_mu = self.MU_BASE
        
        # Track-specific data (placeholder for track geometry)
        self.track_data = self._load_track_data(track_name)
        
    def _load_track_data(self, track_name: str) -> Dict[str, Any]:
        """Load track-specific data (corners, optimal racing line, etc.)"""
        # Placeholder for track data loading
        return {
            "name": track_name,
            "length": 5891,  # meters (e.g., Silverstone)
            "corners": 18,
            "optimal_racing_line": None  # Would contain racing line coordinates
        }
    
    def _pacejka(self, B: float, C: float, D: float, E: float, slip: float) -> float:
        """Compute tire force using Pacejka Magic Formula."""
        return D * np.sin(C * np.arctan(B * slip - E * (B * slip - np.arctan(B * slip))))

    def _calculate_kamm_circle_violation(self, lateral_g: float, brake_pressure: float) -> float:
        """Keep legacy helper for tests; approximate combined slip usage."""
        lateral_force = abs(lateral_g)
        longitudinal_force = abs(brake_pressure)
        return np.sqrt(lateral_force**2 + longitudinal_force**2)

    def _compute_load_transfer(self, ax: float) -> Tuple[float, float]:
        """Dynamic load transfer between axles (front, rear)."""
        static_front = self.total_mass * self.GRAVITY * (self.LR / self.WHEELBASE)
        static_rear = self.total_mass * self.GRAVITY * (self.LF / self.WHEELBASE)
        transfer = (self.total_mass * ax * self.CG_HEIGHT) / self.WHEELBASE
        return static_front + transfer, static_rear - transfer
    
    def _calculate_reward(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        next_state: np.ndarray
    ) -> float:
        """
        Calculate multi-objective reward components.
        
        Returns:
            Total scalar reward combining speed, safety, and smoothness
        """
        velocity = next_state[0]
        lateral_g = next_state[2]
        brake_pressure = next_state[5]
        racing_line_deviation = next_state[6]
        friction_usage = next_state[7]
        
        # Speed reward: Encourage high exit velocity
        r_speed = velocity / self.MAX_VELOCITY
        
        # Safety reward: Penalize exceeding tire friction limits
        if friction_usage > 1.0:
            r_safety = -10.0 * (friction_usage - 1.0)  # Massive penalty for crash risk
        else:
            r_safety = 0.1 * (1.0 - friction_usage)  # Small bonus for staying safe
        
        # Smoothness reward: Penalize abrupt changes in haptic feedback
        # (prevents startling the rider)
        if self.timestep > 0 and hasattr(self, 'last_action'):
            haptic_change = np.linalg.norm(action - self.last_action)
            r_smooth = -0.5 * haptic_change
        else:
            r_smooth = 0.0
        
        # Bonus for staying on racing line
        r_line = -0.1 * abs(racing_line_deviation)
        
        # Combined reward (weighted sum)
        total_reward = 1.0 * r_speed + 2.0 * r_safety + 0.5 * r_smooth + 0.3 * r_line
        
        return total_reward
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to an initial state.
        
        Returns:
            Tuple of (initial_observation, info_dict)
        """
        super().reset(seed=seed)
        
        # Domain randomization for robustness
        mu_jitter = self.np_random.uniform(-self.MU_JITTER, self.MU_JITTER)
        rider_delta = self.np_random.uniform(-self.RIDER_MASS_JITTER, self.RIDER_MASS_JITTER)
        self.surface_mu = max(0.7, self.MU_BASE * (1.0 + mu_jitter))
        self.total_mass = self.BASE_VEHICLE_MASS + self.BASE_RIDER_MASS + rider_delta

        # Initialize dynamic states with mild noise
        self.vx = float(self.np_random.uniform(30.0, 50.0))
        self.vy = float(self.np_random.normal(0.0, 0.5))
        self.yaw_rate = float(self.np_random.normal(0.0, 0.05))
        self.roll_angle_rad = 0.0
        self.roll_rate = 0.0

        # Initialize observable state
        self.state = np.array([
            self.vx,                               # velocity (m/s)
            np.rad2deg(self.roll_angle_rad),       # roll_angle (deg)
            0.0,                                   # lateral_g
            self.np_random.uniform(50.0, 200.0),   # distance_to_apex
            0.55,                                  # throttle_position (auto rider)
            0.0,                                   # brake_pressure
            self.np_random.uniform(-2.0, 2.0),     # racing_line_deviation
            0.2                                    # tire_friction_usage
        ], dtype=np.float32)
        
        self.timestep = 0
        self.last_action = np.zeros(3, dtype=np.float32)
        
        info = {
            "track": self.track_name,
            "lap_time": 0.0
        }
        
        return self.state.copy(), info
    
    def step(
        self, 
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one timestep of the environment.
        
        Args:
            action: Haptic feedback control [left_intensity, right_intensity, frequency]
        
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Clip actions to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Time step (50 Hz integration)
        dt = 0.02

        # Extract observable state
        velocity = self.state[0]
        distance_to_apex = self.state[3]
        throttle = float(self.state[4])
        brake = float(self.state[5])
        
        # Store previous state for reward calculation
        prev_state = self.state.copy()
        
        # Heuristic rider control: reduce throttle near apex, brake harder when close
        throttle = float(np.clip(throttle + self.np_random.normal(0, 0.01), 0.0, 1.0))
        if distance_to_apex < 120:
            brake = float(np.clip(brake + 0.02, 0.0, 1.0))
            throttle *= 0.6
        else:
            brake = float(np.clip(brake - 0.02, 0.0, 1.0))
            throttle = float(np.clip(throttle + 0.02, 0.0, 1.0))

        # Requested longitudinal acceleration
        max_accel = 6.0  # m/s²
        max_decel = 9.0  # m/s²
        ax_request = throttle * max_accel - brake * max_decel

        # Load transfer (front / rear normal loads)
        Fz_f, Fz_r = self._compute_load_transfer(ax_request)
        Fz_f = max(Fz_f, 50.0)
        Fz_r = max(Fz_r, 50.0)

        # Longitudinal slip ratios (approx)
        vx_safe = max(abs(self.vx), 0.5)
        kappa_f = np.clip(ax_request / (vx_safe * 5.0), -0.2, 0.2)
        kappa_r = np.clip(ax_request / (vx_safe * 5.0), -0.2, 0.2)

        # Lateral slip angles
        alpha_f = np.clip(np.arctan2(self.vy + self.LF * self.yaw_rate, vx_safe), -0.6, 0.6)
        alpha_r = np.clip(np.arctan2(self.vy - self.LR * self.yaw_rate, vx_safe), -0.6, 0.6)

        # Pacejka tire forces
        Fx_f = self._pacejka(self.Bx, self.Cx, self.surface_mu * Fz_f, self.Ex, kappa_f)
        Fx_r = self._pacejka(self.Bx, self.Cx, self.surface_mu * Fz_r, self.Ex, kappa_r)
        Fy_f = self._pacejka(self.By, self.Cy, self.surface_mu * Fz_f, self.Ey, alpha_f)
        Fy_r = self._pacejka(self.By, self.Cy, self.surface_mu * Fz_r, self.Ey, alpha_r)

        # Total forces and moments
        Fx_total = Fx_f + Fx_r
        Fy_total = Fy_f + Fy_r
        yaw_moment = Fy_f * self.LF - Fy_r * self.LR

        # 6-DOF simplified dynamics integration
        ax = Fx_total / self.total_mass + self.vy * self.yaw_rate
        ay = Fy_total / self.total_mass - self.vx * self.yaw_rate
        yaw_rate_dot = yaw_moment / self.INERTIA_YAW

        # Roll dynamics: track lean towards centrifugal force
        lateral_acc = ay + self.vx * self.yaw_rate
        roll_eq = np.arctan2(lateral_acc, self.GRAVITY)  # radians
        roll_tau = 0.3
        roll_error = roll_eq - self.roll_angle_rad
        roll_acc = roll_error / roll_tau
        self.roll_rate += roll_acc * dt
        self.roll_angle_rad += self.roll_rate * dt
        self.roll_angle_rad = np.clip(self.roll_angle_rad, -np.deg2rad(self.MAX_LEAN_ANGLE), np.deg2rad(self.MAX_LEAN_ANGLE))

        # Integrate linear and yaw states
        self.vx = float(np.clip(self.vx + ax * dt, 0.0, self.MAX_VELOCITY))
        self.vy = float(self.vy + ay * dt)
        self.yaw_rate = float(self.yaw_rate + yaw_rate_dot * dt)

        # Update distance to apex along path
        distance_to_apex = max(0.0, distance_to_apex - self.vx * dt)

        # Lateral g-load
        lateral_g = float(np.clip(lateral_acc / self.GRAVITY, -self.MAX_LATERAL_G, self.MAX_LATERAL_G))

        # Friction usage (normalized combined force)
        friction_usage_raw = float(np.sqrt((Fx_total / (self.surface_mu * self.total_mass * self.GRAVITY))**2 + (Fy_total / (self.surface_mu * self.total_mass * self.GRAVITY))**2))
        friction_usage_raw = float(np.clip(friction_usage_raw, 0.0, 1.8))
        friction_usage = float(np.clip(friction_usage_raw, 0.0, 1.0))

        # Racing line deviation with noise (simulate rider precision)
        racing_line_deviation = float(np.clip(self.state[6] + self.np_random.normal(0, 0.3) + self.yaw_rate * 0.05, -10.0, 10.0))

        # Update observable state
        self.state = np.array([
            self.vx,
            np.rad2deg(self.roll_angle_rad),
            lateral_g,
            distance_to_apex,
            throttle,
            brake,
            racing_line_deviation,
            friction_usage
        ], dtype=np.float32)
        
        # Calculate reward based on state transition
        reward = self._calculate_reward(prev_state, action, self.state)
        
        # Check termination conditions
        terminated = False
        if friction_usage_raw > 1.2:
            terminated = True  # Crash
            reward -= 50.0
        
        # Check truncation (max timesteps)
        self.timestep += 1
        truncated = self.timestep >= self.max_timesteps
        
        # Info dict
        info = {
            "velocity": float(velocity),
            "friction_usage": float(friction_usage),
            "friction_usage_raw": float(friction_usage_raw),
            "safety_margin": float(1.0 - friction_usage_raw),
            "reward": float(reward)
        }
        
        self.last_action = action.copy()
        
        return self.state.copy(), float(reward), terminated, truncated, info
    
    def render(self):
        """Render the environment (placeholder)."""
        if self.render_mode == "human":
            print(f"Step {self.timestep}: v={self.state[0]:.1f} m/s, "
                  f"roll={self.state[1]:.1f}°, "
                  f"friction={self.state[7]:.2f}")
        return None
    
    def close(self):
        """Clean up environment resources."""
        pass


# Register environment with Gymnasium
gym.register(
    id='MotorcycleRacing-v0',
    entry_point='motorcycle_env:MotorcycleEnv',
    max_episode_steps=1000,
)
