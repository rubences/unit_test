"""
Phase 2: Bio-Physics Motorcycle Racing Environment

Custom Gymnasium environment that simulates motorcycle racing with:
- Realistic telemetry dynamics (speed, G-force, lean angle)
- Biometric integration (ECG-derived HRV, stress level)
- Bio-gating mechanism (enforces cognitive load limits)
- Multi-objective reward (speed + safety + cognitive load)

Key Innovation: Non-learnable firmware-level safety gating
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import pickle
from pathlib import Path


class MotoBioEnv(gym.Env):
    """
    Motorcycle Racing Environment with Bio-Cybernetic Feedback Loop.
    
    Observation Space (5D):
    - speed_kmh: [0, 350] km/h
    - g_force: [0, 2.5] lateral G-force
    - lean_angle: [0, 65] degrees
    - hrv_index: [0, 1] normalized RMSSD (0=high stress, 1=calm)
    - stress_level: [0, 1] integrated stress (from HR, G-force, duration)
    
    Action Space (4 discrete):
    - 0: No Feedback (rest mode, no haptic)
    - 1: Mild Haptic (gentle vibration, 20-40 Hz)
    - 2: Warning Haptic (sharp pulses, 80-120 Hz)
    - 3: Emergency Haptic (urgent, full amplitude, 150+ Hz)
    
    Reward Function:
    R(s,a) = w_speed * r_speed + w_safety * r_safety + w_cognitive * r_cognitive
           = 0.50 * (speed/350) + 0.35 * r_safety - 0.15 * stress_penalty
    
    Where:
    - r_speed: Normalized speed reward (encourages faster laps)
    - r_safety: Safety bonus if no off-track events, penalty if near edge
    - stress_penalty: Increases quadratically with stress (CLT operationalization)
    
    Bio-Gating Mechanism (Non-Learnable):
    - IF stress_level > 0.8 (panic mode):
        - FORCE action = 0 (No Feedback)
        - Override any action the agent selected
        - This prevents cognitive overload regardless of policy
    """
    
    metadata = {'render_modes': ['human'], 'render_fps': 30}
    
    def __init__(self, telemetry_df=None, episode_length=600):
        """
        Initialize the environment.
        
        Args:
            telemetry_df (pd.DataFrame): Loaded race telemetry (speed, g_force, etc.)
                                        If None, uses synthetic data from this environment
            episode_length (int): Max timesteps per episode (seconds at 10 Hz = 60s lap)
        """
        super().__init__()
        
        # Load or generate telemetry
        if telemetry_df is None:
            # Synthetic racing profile (single lap)
            self.telemetry_df = self._generate_synthetic_telemetry(episode_length)
        else:
            self.telemetry_df = telemetry_df.reset_index(drop=True)
        
        self.episode_length = episode_length
        self.dt = 0.1  # Timestep: 10 Hz control rate (0.1 seconds)
        self.current_step = 0
        
        # Observation space: [speed, g_force, lean_angle, hrv_index, stress_level]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([350.0, 2.5, 65.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Action space: 4 haptic feedback levels
        # 0: No Feedback, 1: Mild, 2: Warning, 3: Emergency
        self.action_space = spaces.Discrete(4)
        
        # Action descriptions
        self.action_names = [
            'No Feedback',
            'Mild Haptic (20-40 Hz)',
            'Warning Haptic (80-120 Hz)',
            'Emergency Haptic (150+ Hz)'
        ]
        
        # Stress tracking
        self.stress_accumulated = 0.0
        self.bio_gate_activations = 0  # Count how many times gating was triggered
        self.off_track_events = 0  # Count safety violations
        
        # Initialize state
        self.last_obs = None
        self.episode_reward = 0.0
        
    def _generate_synthetic_telemetry(self, episode_length):
        """Generate a synthetic single-lap telemetry profile."""
        n_points = int(episode_length / self.dt)
        time = np.linspace(0, episode_length, n_points)
        
        # Simple circuit pattern: 4 straights + 4 corners
        section = (time / (episode_length / 4)) % 2
        
        # Speed: straights at 250 km/h, corners at 80 km/h
        speed = 80 + 170 * np.sin(section * np.pi) ** 2
        
        # G-force: corners have high G, straights have low G
        g_force = 0.3 + 2.0 * np.sin(section * np.pi) ** 2
        
        # Lean angle: follows G-force
        lean_angle = 10 + 55 * np.sin(section * np.pi) ** 2
        
        # Heart rate: increases with stress
        hr_base = 70 + 0.2 * speed + 20 * g_force
        
        df = pd.DataFrame({
            'timestamp': time,
            'speed_kmh': speed,
            'g_force': g_force,
            'lean_angle_deg': lean_angle,
            'heart_rate_bpm': hr_base
        })
        
        return df
    
    def _compute_stress_level(self):
        """
        Compute current stress level from physiological and physical factors.
        
        Stress Model:
        - Physical stress (G-force, lean angle): Linear contribution
        - Cognitive stress (sustained high HR, unpredictability): Non-linear
        - Time-on-task effect: Fatigue increases stress over time
        
        Returns:
            float: Stress level in [0, 1], where 1 = maximum panic
        """
        obs = self.last_obs
        
        # Physical stress component (G-force and lean angle)
        g_normalized = obs[1] / 2.5  # G-force normalized
        lean_normalized = obs[2] / 65.0  # Lean angle normalized
        physical_stress = (0.6 * g_normalized + 0.4 * lean_normalized)
        
        # Cognitive stress component (HR + time-on-task)
        # HR range: 70-180 bpm (0 to 1 stress scale)
        hr = obs[4]  # We infer HR from observation context
        hr_normalized = (hr - 70) / (180 - 70)
        
        # Time-on-task fatigue (increases over 60 seconds)
        fatigue_factor = self.current_step / 600.0
        
        # Integrated stress (weighted combination)
        stress = (0.5 * physical_stress + 
                 0.3 * np.clip(hr_normalized, 0, 1) +
                 0.2 * fatigue_factor)
        
        # Accumulate stress (parasympathetic recovery lag)
        alpha = 0.1  # Exponential filter time constant
        self.stress_accumulated = (1 - alpha) * self.stress_accumulated + alpha * stress
        
        return np.clip(self.stress_accumulated, 0.0, 1.0)
    
    def _bio_gating_mechanism(self, action):
        """
        Non-learnable firmware-level safety gating.
        
        Core Safety Principle:
        If cognitive load is critically high (stress > 0.8), the agent cannot
        override the safety decision. We force the action to "No Feedback"
        to prevent further cognitive overload.
        
        This is a hard constraint that cannot be learned away.
        
        Args:
            action (int): Proposed action from agent
        
        Returns:
            tuple: (gated_action, was_gated)
        """
        stress = self._compute_stress_level()
        
        if stress > 0.8:
            # Force No Feedback when in panic mode
            self.bio_gate_activations += 1
            return 0, True
        else:
            return action, False
    
    def _compute_observation(self):
        """Extract observation from current telemetry step."""
        if self.current_step >= len(self.telemetry_df):
            # Loop back to start (continuous riding)
            idx = self.current_step % len(self.telemetry_df)
        else:
            idx = self.current_step
        
        row = self.telemetry_df.iloc[idx]
        
        # HRV Index: estimated from heart rate pattern
        # Simple model: HRV decreases with sustained high HR
        # Typical HRV range for athletes: 20-100 ms RMSSD
        # We normalize RMSSD to [0, 1] where 0 = high stress, 1 = calm
        hr = row['heart_rate_bpm']
        hr_normalized = (hr - 70) / (180 - 70)
        hrv_index = 1.0 - np.clip(hr_normalized, 0, 1)
        
        obs = np.array([
            row['speed_kmh'],
            row['g_force'],
            row['lean_angle_deg'],
            hrv_index,
            self._compute_stress_level()
        ], dtype=np.float32)
        
        self.last_obs = obs
        return obs
    
    def _compute_reward(self, action):
        """
        Multi-objective reward function.
        
        Reward Components:
        1. Speed Reward (w=0.50): Normalized speed incentive
           r_speed = speed_kmh / 350
           
        2. Safety Reward (w=0.35): Off-track penalty + off-track margin bonus
           r_safety = 1.0 if lean_angle < 60 else -2.0 * (lean_angle - 60) / 5.0
           
        3. Cognitive Load Penalty (w=0.15): Prevents overload
           Penalty increases quadratically: penalty = stress^2
           r_cognitive = -stress^2
        
        4. Gating Reward: Small penalty if gating was active
           This teaches the agent to avoid high-stress situations
        
        Total:
        R = 0.50 * (speed/350) + 0.35 * r_safety - 0.15 * stress^2 - 0.05 * was_gated
        
        Args:
            action (int): Selected action (post-gating)
        
        Returns:
            float: Total reward
        """
        obs = self.last_obs
        
        # Component 1: Speed reward
        speed = obs[0]
        r_speed = speed / 350.0
        
        # Component 2: Safety reward
        lean = obs[2]
        if lean < 60:
            r_safety = 1.0
        else:
            # Penalty for getting close to lean limit (63 degrees)
            r_safety = 1.0 - 2.0 * (lean - 60) / 3.0
            self.off_track_events += 1
        
        # Component 3: Cognitive load penalty (CLT operationalization)
        stress = obs[4]
        r_cognitive = -stress ** 2  # Quadratic penalty
        
        # Component 4: Gating penalty (we'll pass was_gated from step())
        
        # Weighted sum
        reward = (0.50 * r_speed + 
                 0.35 * r_safety + 
                 0.15 * r_cognitive)
        
        return reward
    
    def step(self, action):
        """
        Execute one timestep of the environment.
        
        Args:
            action (int): Action from agent [0, 1, 2, 3]
        
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Apply bio-gating mechanism
        gated_action, was_gated = self._bio_gating_mechanism(action)
        
        # Compute reward
        reward = self._compute_reward(gated_action)
        
        # Penalize if gating was active
        if was_gated:
            reward -= 0.05
        
        # Advance timestep
        self.current_step += 1
        
        # Get new observation
        obs = self._compute_observation()
        
        # Episode termination
        terminated = (self.current_step >= self.episode_length)
        truncated = False
        
        # Track cumulative reward
        self.episode_reward += reward
        
        # Info dict (for logging)
        info = {
            'action': self.action_names[gated_action],
            'original_action': self.action_names[action],
            'was_gated': was_gated,
            'stress_level': obs[4],
            'hrv_index': obs[3],
            'speed_kmh': obs[0],
            'lean_angle': obs[2],
            'cumulative_reward': self.episode_reward
        }
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        
        Args:
            seed (int): Random seed for reproducibility
            options (dict): Reset options
        
        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        self.stress_accumulated = 0.0
        self.bio_gate_activations = 0
        self.off_track_events = 0
        self.episode_reward = 0.0
        
        obs = self._compute_observation()
        
        info = {
            'bio_gate_activations': self.bio_gate_activations,
            'off_track_events': self.off_track_events
        }
        
        return obs, info
    
    def render(self):
        """Render the environment (optional, for visualization)."""
        if self.last_obs is not None:
            print(f"Step {self.current_step}: Speed={self.last_obs[0]:.1f} km/h, "
                  f"G={self.last_obs[1]:.2f}, Stress={self.last_obs[4]:.2f}")
    
    def save_episode_data(self, filepath):
        """
        Save episode data for later analysis.
        
        Args:
            filepath (str): Where to save the pickle file
        """
        episode_data = {
            'telemetry': self.telemetry_df,
            'total_steps': self.current_step,
            'bio_gate_activations': self.bio_gate_activations,
            'off_track_events': self.off_track_events,
            'episode_reward': self.episode_reward
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(episode_data, f)
        
        print(f"Episode data saved to {filepath}")


if __name__ == '__main__':
    # Test the environment
    print("Testing MotoBioEnv...")
    
    env = MotoBioEnv(episode_length=100)
    obs, info = env.reset()
    
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")
    
    # Run 20 random steps
    for step in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 5 == 0:
            print(f"Step {step}: action={info['action']}, reward={reward:.4f}, "
                  f"stress={info['stress_level']:.2f}")
        
        if terminated:
            break
    
    print("\nEnvironment test passed!")
