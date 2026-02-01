"""
Custom Gymnasium Environment with Bio-Gating Mechanism
Implements the POMDP for bio-adaptive haptic coaching from the research paper
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from typing import Tuple, Dict, Any
from pathlib import Path

# Soporte para importación relativa y absoluta
try:
    from .config import SIM_CONFIG, REWARD_CONFIG, PATHS
except ImportError:
    from config import SIM_CONFIG, REWARD_CONFIG, PATHS


class MotoBioEnv(gym.Env):
    """
    Motorcycle Racing Environment with Bio-Gating Safety Override
    
    State: [Speed, Lean Angle, G-Force, HRV Index, Stress Level]
    Actions: [No Feedback, Mild Haptic, Warning Haptic, Emergency Haptic]
    
    Bio-Gating Rule: IF stress > PANIC_THRESHOLD, force action=0 (no feedback)
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, telemetry_df: pd.DataFrame = None):
        """
        Initialize motorcycle racing environment
        
        Args:
            telemetry_df: Pandas dataframe with prerecorded telemetry
        """
        super().__init__()
        
        self.telemetry_df = telemetry_df
        self.current_step = 0
        self.episode_steps = 0
        self.off_track_count = 0
        self.bio_gate_activations = 0
        self.episode_stress_integral = 0.0
        
        # Observation space: [speed, lean_angle, g_force, hrv_index, stress_level]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([
                SIM_CONFIG.MAX_SPEED_KMH,
                SIM_CONFIG.MAX_LEAN_ANGLE,
                SIM_CONFIG.MAX_G_FORCE,
                1.0,  # HRV normalized
                1.0,  # Stress normalized
            ]),
            dtype=np.float32
        )
        
        # Action space: [No Feedback=0, Mild=1, Warning=2, Emergency=3]
        self.action_space = spaces.Discrete(4)
        
        # Action names for debugging
        self.action_names = ["No Feedback", "Mild Haptic", "Warning Haptic", "Emergency Haptic"]
    
    def _get_telemetry(self, step: int) -> Dict[str, float]:
        """
        Get telemetry data for current step
        
        Args:
            step: Current step index
            
        Returns:
            Dictionary with speed, lean_angle, g_force, heart_rate
        """
        if self.telemetry_df is None or step >= len(self.telemetry_df):
            # Fallback: generate synthetic data
            return {
                'speed_kmh': np.random.uniform(100, 300),
                'lean_angle_deg': np.random.uniform(0, 65),
                'g_force': np.random.uniform(0, 2.5),
                'heart_rate_bpm': np.random.uniform(80, 180),
            }
        
        row = self.telemetry_df.iloc[step]
        return {
            'speed_kmh': float(row['speed_kmh']),
            'lean_angle_deg': float(row['lean_angle_deg']),
            'g_force': float(row['g_force']),
            'heart_rate_bpm': float(row['heart_rate_bpm']),
        }
    
    def _compute_stress_level(self, heart_rate: float, on_track: bool = True) -> float:
        """
        Compute normalized stress level (0.0 to 1.0) from physiology
        
        High HR relative to max = high stress
        Also increases with time-on-task
        
        Args:
            heart_rate: Current heart rate in bpm
            on_track: Whether driver is on track
            
        Returns:
            Normalized stress level (0.0-1.0)
        """
        # Physiological stress from HR
        hr_stress = (heart_rate - SIM_CONFIG.RESTING_HR) / (SIM_CONFIG.MAX_HR - SIM_CONFIG.RESTING_HR)
        hr_stress = np.clip(hr_stress, 0.0, 1.0)
        
        # Mental fatigue: increases with episode duration
        fatigue = min(self.episode_steps / (REWARD_CONFIG.EPISODE_STEPS_MAX * 0.5), 1.0)
        
        # Off-track penalty adds stress
        off_track_stress = 0.3 if not on_track else 0.0
        
        stress = 0.6 * hr_stress + 0.2 * fatigue + 0.2 * off_track_stress
        stress = np.clip(stress, 0.0, 1.0)
        
        return stress
    
    def _bio_gating_mechanism(self, action: int, stress: float) -> Tuple[int, bool]:
        """
        Bio-Gating Override Mechanism (from paper)
        
        Non-learnable firmware-level safety constraint:
        IF stress > PANIC_THRESHOLD, FORCE action = 0 (No Feedback)
        
        This prevents the RL policy from overwhelming a panicked driver
        
        Args:
            action: Proposed action from policy
            stress: Current stress level (0.0-1.0)
            
        Returns:
            Tuple of (final_action, was_gated)
        """
        was_gated = False
        
        if stress > SIM_CONFIG.PANIC_THRESHOLD:
            # Override: force no feedback
            final_action = 0
            was_gated = True
            self.bio_gate_activations += 1
        else:
            final_action = action
        
        return final_action, was_gated
    
    def _compute_reward(self, speed: float, lean_angle: float, stress: float, 
                       on_track: bool, action: int, gated: bool) -> float:
        """
        Multi-objective reward function (from paper)
        
        Reward = 0.50*speed + 0.35*safety - 0.15*stress²
        
        Args:
            speed: Speed in km/h
            lean_angle: Lean angle in degrees
            stress: Normalized stress (0.0-1.0)
            on_track: Whether on track
            action: Action taken
            gated: Whether bio-gating was activated
            
        Returns:
            Reward signal
        """
        # Speed reward (normalized to 0-1)
        speed_normalized = speed / REWARD_CONFIG.SPEED_NORMALIZATION_FACTOR
        speed_reward = speed_normalized * REWARD_CONFIG.SPEED_WEIGHT
        
        # Safety reward: penalize off-track and extreme lean
        if not on_track:
            safety_reward = REWARD_CONFIG.OFF_TRACK_PENALTY
        else:
            lean_penalty = max(0, (lean_angle - REWARD_CONFIG.LEAN_ANGLE_SAFE_MAX) / 10.0)
            safety_reward = (1.0 - lean_penalty) * REWARD_CONFIG.SAFETY_WEIGHT
        
        # Stress penalty: penalize high stress
        stress_penalty = -(stress ** 2) * REWARD_CONFIG.STRESS_PENALTY_WEIGHT
        
        # Bonus for effective bio-gating (letting the system help during panic)
        bio_gate_bonus = 0.5 if (gated and action != 0) else 0.0
        
        total_reward = speed_reward + safety_reward + stress_penalty + bio_gate_bonus
        
        return float(total_reward)
    
    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        self.episode_steps = 0
        self.off_track_count = 0
        self.bio_gate_activations = 0
        self.episode_stress_integral = 0.0
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation state
        
        Returns:
            5D observation vector
        """
        telem = self._get_telemetry(self.current_step)
        
        speed_norm = telem['speed_kmh'] / SIM_CONFIG.MAX_SPEED_KMH
        lean_norm = telem['lean_angle_deg'] / SIM_CONFIG.MAX_LEAN_ANGLE
        g_force_norm = telem['g_force'] / SIM_CONFIG.MAX_G_FORCE
        
        # HRV index (simplified: inverse of HR)
        hrv_index = 1.0 - (telem['heart_rate_bpm'] - SIM_CONFIG.RESTING_HR) / (SIM_CONFIG.MAX_HR - SIM_CONFIG.RESTING_HR)
        hrv_index = np.clip(hrv_index, 0.0, 1.0)
        
        # Stress level
        stress = self._compute_stress_level(telem['heart_rate_bpm'])
        
        obs = np.array([
            speed_norm,
            lean_norm,
            g_force_norm,
            hrv_index,
            stress,
        ], dtype=np.float32)
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Return info dictionary"""
        telem = self._get_telemetry(self.current_step)
        
        return {
            "speed_kmh": telem['speed_kmh'],
            "heart_rate_bpm": telem['heart_rate_bpm'],
            "bio_gate_activations": self.bio_gate_activations,
            "off_track_count": self.off_track_count,
        }
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step
        
        Args:
            action: Action from policy (0-3)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.episode_steps += 1
        
        # Get current telemetry
        telem = self._get_telemetry(self.current_step)
        speed = telem['speed_kmh']
        lean_angle = telem['lean_angle_deg']
        heart_rate = telem['heart_rate_bpm']
        
        # Compute stress
        stress = self._compute_stress_level(heart_rate)
        self.episode_stress_integral += stress
        
        # Check if on track (simplified: based on lean angle)
        on_track = lean_angle < SIM_CONFIG.MAX_LEAN_ANGLE
        if not on_track:
            self.off_track_count += 1
        
        # Apply bio-gating mechanism
        final_action, was_gated = self._bio_gating_mechanism(action, stress)
        
        # Compute reward
        reward = self._compute_reward(speed, lean_angle, stress, on_track, final_action, was_gated)
        
        # Termination conditions
        terminated = (not on_track) or (stress > 0.95)  # Crash or panic
        truncated = self.episode_steps >= REWARD_CONFIG.EPISODE_STEPS_MAX  # Time limit
        
        # Move to next step
        self.current_step += 1
        if self.telemetry_df is not None:
            if self.current_step >= len(self.telemetry_df):
                truncated = True
        
        # Get next observation
        obs = self._get_observation()
        info = self._get_info()
        info["action_taken"] = self.action_names[final_action]
        info["was_bio_gated"] = was_gated
        
        return obs, reward, terminated, truncated, info
    
    def render(self) -> None:
        """Render environment (placeholder)"""
        pass
    
    def get_episode_summary(self) -> Dict[str, float]:
        """Get summary statistics for current episode"""
        return {
            "total_steps": float(self.episode_steps),
            "off_track_count": float(self.off_track_count),
            "bio_gate_activations": float(self.bio_gate_activations),
            "avg_stress": float(self.episode_stress_integral / max(1, self.episode_steps)),
        }


def main():
    """Test environment"""
    from .data_gen import SyntheticTelemetry
    
    # Generate sample data
    gen = SyntheticTelemetry()
    session = gen.generate_race_session(n_laps=5)
    
    # Create environment
    env = MotoBioEnv(telemetry_df=session.telemetry_df)
    
    print("🧪 Testing MotoBioEnv...")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Run a few steps
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step}: Action={info['action_taken']}, Reward={reward:.3f}, Speed={info['speed_kmh']:.1f} km/h")
        
        if terminated or truncated:
            break
    
    print("✅ Environment test complete!")


if __name__ == "__main__":
    main()
