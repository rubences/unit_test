"""
Synthetic Motorcycle Racing Data Generation for Minari Dataset

This module generates synthetic motorcycle racing telemetry data following Minari's
Dataset Creation API. It simulates two rider profiles:
    1. Pro Rider: Optimal braking, smooth inputs, apex optimization
    2. Amateur Rider: Inconsistent braking, rough inputs, variable line choice

The generated dataset is saved in Minari format (.hdf5) for use in offline RL training.

Reference: Minari Documentation - https://minari.readthedocs.io/
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import json
from datetime import datetime
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProRiderModel:
    """
    Professional rider model with optimal racing characteristics.
    
    Characteristics:
    - Smooth throttle and brake inputs
    - Optimal apex speed (high lean angle + velocity)
    - Minimal racing line deviation
    - Consistent braking points
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize pro rider model.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
        self.braking_points = {}  # Cache for consistent braking points
        
    def compute_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute rider action based on current state.
        
        Pro rider logic:
        - Aggressive braking (0.8-1.0) near apex (distance < 50m)
        - Smooth throttle application (0.7-1.0) on straights
        - Maintains high lean angles (40-55 degrees) in corners
        
        Args:
            observation: Current observation from environment
            
        Returns:
            Action array [haptic_left, haptic_right, haptic_freq]
        """
        velocity = observation[0]
        distance_to_apex = observation[3]
        throttle_pos = observation[4]
        brake_pressure = observation[5]
        lean_angle = observation[1]
        
        # Pro rider: Strong haptic feedback in critical phases
        haptic_left = 0.0
        haptic_right = 0.0
        haptic_freq = 150.0
        
        # Braking phase: high-frequency haptic on both gloves
        if distance_to_apex < 50 and velocity > 30:
            haptic_left = 0.9
            haptic_right = 0.9
            haptic_freq = 250.0  # High frequency for braking urgency
        
        # Cornering phase: asymmetric haptic based on lean angle
        elif lean_angle > 35:
            if lean_angle > 0:  # Right lean
                haptic_right = 0.7
                haptic_left = 0.3
            else:  # Left lean
                haptic_left = 0.7
                haptic_right = 0.3
            haptic_freq = 180.0
        
        # Acceleration phase: light feedback
        else:
            haptic_left = 0.2
            haptic_right = 0.2
            haptic_freq = 100.0
        
        return np.array([haptic_left, haptic_right, haptic_freq], dtype=np.float32)


class AmateurRiderModel:
    """
    Amateur rider model with suboptimal racing characteristics.
    
    Characteristics:
    - Jerky throttle and brake inputs (noisy)
    - Suboptimal apex speed
    - Variable racing line (±5m deviation)
    - Inconsistent braking timing
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize amateur rider model.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
        self.input_noise = 0.15  # Noise std dev for inputs
        
    def compute_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute rider action with amateur characteristics.
        
        Amateur rider logic:
        - Late or early braking (50-100m from apex)
        - Unsmooth throttle inputs (jerky)
        - Random haptic feedback (less coherent)
        
        Args:
            observation: Current observation from environment
            
        Returns:
            Action array [haptic_left, haptic_right, haptic_freq]
        """
        velocity = observation[0]
        distance_to_apex = observation[3]
        lean_angle = observation[1]
        
        # Amateur rider: inconsistent, noisy inputs
        haptic_left = self.rng.uniform(0.0, 0.8)
        haptic_right = self.rng.uniform(0.0, 0.8)
        haptic_freq = self.rng.uniform(80, 280)
        
        # Add some structure but with noise
        if distance_to_apex < 80:
            haptic_left += self.rng.normal(0.2, self.input_noise)
            haptic_right += self.rng.normal(0.2, self.input_noise)
        
        # Clamp to valid ranges
        haptic_left = np.clip(haptic_left, 0.0, 1.0)
        haptic_right = np.clip(haptic_right, 0.0, 1.0)
        haptic_freq = np.clip(haptic_freq, 50.0, 300.0)
        
        return np.array([haptic_left, haptic_right, haptic_freq], dtype=np.float32)


class MotorcycleEnvWrapper:
    """
    Lightweight motorcycle environment for data generation.
    
    This is a minimal environment for generating realistic motorcycle racing
    telemetry without requiring the full Gymnasium environment.
    """
    
    def __init__(self, lap_length: int = 300, seed: int = 42):
        """
        Initialize the environment wrapper.
        
        Args:
            lap_length: Number of timesteps per lap (~5 seconds at 60Hz)
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
        self.lap_length = lap_length
        self.timestep = 0
        self.lap_number = 0
        self.velocity = 20.0  # Initial velocity m/s
        self.lean_angle = 0.0
        self.lateral_g = 0.0
        self.distance_to_apex = 250.0
        self.throttle_position = 0.5
        self.brake_pressure = 0.0
        self.racing_line_deviation = 0.0
        self.tire_friction_usage = 0.3
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.timestep = 0
        self.velocity = 20.0
        self.lean_angle = 0.0
        self.lateral_g = 0.0
        self.distance_to_apex = 250.0
        self.throttle_position = 0.5
        self.brake_pressure = 0.0
        self.racing_line_deviation = 0.0
        self.tire_friction_usage = 0.3
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Step the environment forward with haptic action.
        
        Args:
            action: Haptic action [left_intensity, right_intensity, frequency]
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Simulate physics based on previous state
        self._update_dynamics()
        
        # Calculate reward (negative lap time - safety violations)
        reward = self._compute_reward()
        
        # Check termination conditions
        terminated = self.velocity < 5.0  # Too slow
        truncated = self.timestep >= self.lap_length  # Lap complete
        
        info = {
            'lap_number': self.lap_number,
            'lap_time': self.timestep / 60.0,  # Convert to seconds (60Hz)
            'velocity': float(self.velocity),
            'action_taken': action.tolist()
        }
        
        self.timestep += 1
        if truncated:
            self.lap_number += 1
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        return np.array([
            self.velocity,
            self.lean_angle,
            self.lateral_g,
            self.distance_to_apex,
            self.throttle_position,
            self.brake_pressure,
            self.racing_line_deviation,
            self.tire_friction_usage
        ], dtype=np.float32)
    
    def _update_dynamics(self):
        """Update environment dynamics (simplified physics)."""
        # Distance to apex decreases, wraps at 0 (lap reset)
        self.distance_to_apex -= 1.0
        if self.distance_to_apex <= 0:
            self.distance_to_apex = 300.0
        
        # Velocity dynamics: accelerate on straights, decelerate in corners
        if self.distance_to_apex > 100:
            # Straight section: accelerate
            self.velocity += self.rng.normal(0.5, 0.2)
            self.velocity = np.clip(self.velocity, 20.0, 95.0)
            self.lean_angle = 0.0
        else:
            # Corner section: decelerate based on position
            decel = 1.0 - (self.distance_to_apex / 100.0)
            self.velocity -= decel * self.rng.normal(1.0, 0.3)
            self.velocity = np.clip(self.velocity, 10.0, 80.0)
            # Lean angle increases in corners
            self.lean_angle = 45.0 * (1.0 - self.distance_to_apex / 100.0)
            self.lean_angle += self.rng.normal(0, 2.0)
            self.lean_angle = np.clip(self.lean_angle, -60.0, 60.0)
        
        # Lateral G forces from lean angle
        self.lateral_g = (self.lean_angle / 60.0) * 2.0
        self.lateral_g += self.rng.normal(0, 0.1)
        
        # Tire friction usage
        self.tire_friction_usage = abs(self.lateral_g) / 2.0 + (abs(self.brake_pressure) * 0.3)
        self.tire_friction_usage = np.clip(self.tire_friction_usage, 0.0, 1.0)
        
        # Racing line deviation
        self.racing_line_deviation += self.rng.normal(0, 0.5)
        self.racing_line_deviation = np.clip(self.racing_line_deviation, -10.0, 10.0)
        
        # Throttle position dynamics
        self.throttle_position += self.rng.normal(0, 0.05)
        self.throttle_position = np.clip(self.throttle_position, 0.0, 1.0)
        
        # Brake pressure dynamics
        self.brake_pressure = max(0.0, self.brake_pressure - 0.02)
    
    def _compute_reward(self) -> float:
        """
        Compute reward: negative lap time + safety bonus.
        
        Reward factors:
        - Negative lap time penalty (faster = better)
        - Safety bonus if no violations (lateral_g < 1.8, tire_friction < 0.95)
        """
        base_reward = -(self.timestep / 60.0) / 100.0  # Normalized lap time
        
        # Safety bonus
        safety_violation = 0.0
        if abs(self.lateral_g) > 1.8:
            safety_violation -= 0.1
        if self.tire_friction_usage > 0.95:
            safety_violation -= 0.05
        
        return base_reward + safety_violation


def generate_minari_episodes(
    env: MotorcycleEnvWrapper,
    rider_model,
    num_laps: int = 100,
    seed: int = 42,
    rider_name: str = "pro"
) -> List[Dict]:
    """
    Generate episodes following Minari Dataset format.
    
    Minari Episode Structure:
    {
        'observations': [obs0, obs1, ...],
        'actions': [action0, action1, ...],
        'rewards': [reward0, reward1, ...],
        'truncations': [False, False, ..., True],
        'terminations': [False, False, ...],
        'infos': [{...}, {...}, ...],
        'episode_id': 'episode_0'
    }
    
    Args:
        env: Environment wrapper
        rider_model: Rider model (Pro or Amateur)
        num_laps: Number of laps to generate
        seed: Random seed
        rider_name: Name of rider profile
        
    Returns:
        List of episodes in Minari format
    """
    episodes = []
    
    logger.info(f"Generating {num_laps} laps for {rider_name} rider...")
    
    for lap_idx in range(num_laps):
        observations = []
        actions = []
        rewards = []
        truncations = []
        terminations = []
        infos = []
        
        obs, _ = env.reset()
        observations.append(obs)
        
        done = False
        step_count = 0
        max_steps = env.lap_length
        
        while not done and step_count < max_steps:
            try:
                # Compute action from rider model
                action = rider_model.compute_action(obs)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Record transition
                actions.append(action)
                rewards.append(reward)
                terminations.append(terminated)
                truncations.append(truncated)
                infos.append(info)
                observations.append(obs)
                
                done = terminated or truncated
                step_count += 1
                
            except Exception as e:
                logger.error(f"Error during step {step_count} of lap {lap_idx}: {e}")
                raise
        
        # Create Minari episode
        episode = {
            'episode_id': f'{rider_name}_episode_{lap_idx}',
            'observations': np.array(observations, dtype=np.float32),
            'actions': np.array(actions, dtype=np.float32),
            'rewards': np.array(rewards, dtype=np.float32),
            'truncations': np.array(truncations, dtype=bool),
            'terminations': np.array(terminations, dtype=bool),
            'infos': infos
        }
        
        episodes.append(episode)
        
        if (lap_idx + 1) % 20 == 0:
            avg_reward = np.mean(episode['rewards'])
            logger.info(f"  Lap {lap_idx + 1}/{num_laps} - Avg Reward: {avg_reward:.4f}")
    
    logger.info(f"Successfully generated {len(episodes)} episodes")
    return episodes


def save_to_minari_hdf5(episodes: List[Dict], output_path: str, rider_name: str):
    """
    Save episodes to HDF5 format compatible with Minari.
    
    Note: This uses h5py directly for maximum compatibility. For full Minari
    integration, consider using minari.create_dataset_from_dict_list() in production.
    
    Args:
        episodes: List of episodes in Minari format
        output_path: Path to save HDF5 file
        rider_name: Name of rider profile for metadata
    """
    try:
        import h5py
    except ImportError:
        logger.error("h5py is required. Install with: pip install h5py")
        raise
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving {len(episodes)} episodes to {output_path}")
    
    with h5py.File(output_path, 'w') as hf:
        # Create metadata group
        metadata = hf.create_group('metadata')
        metadata.attrs['dataset_name'] = f'motorcycle_racing_{rider_name}'
        metadata.attrs['rider_profile'] = rider_name
        metadata.attrs['algorithm'] = 'PPO'
        metadata.attrs['observation_space_dim'] = 8
        metadata.attrs['action_space_dim'] = 3
        metadata.attrs['creation_date'] = datetime.now().isoformat()
        metadata.attrs['total_episodes'] = len(episodes)
        
        # Create episodes group
        episodes_group = hf.create_group('episodes')
        
        for episode_idx, episode in enumerate(episodes):
            ep_group = episodes_group.create_group(f"episode_{episode_idx}")
            
            # Store arrays
            ep_group.create_dataset('observations', data=episode['observations'], compression='gzip')
            ep_group.create_dataset('actions', data=episode['actions'], compression='gzip')
            ep_group.create_dataset('rewards', data=episode['rewards'], compression='gzip')
            ep_group.create_dataset('truncations', data=episode['truncations'], compression='gzip')
            ep_group.create_dataset('terminations', data=episode['terminations'], compression='gzip')
            
            # Store episode info as JSON
            ep_info = {
                'episode_id': episode['episode_id'],
                'num_steps': len(episode['actions']),
                'total_reward': float(np.sum(episode['rewards'])),
                'avg_reward': float(np.mean(episode['rewards'])),
                'infos': episode['infos']
            }
            ep_group.attrs['episode_info'] = json.dumps(ep_info)
    
    logger.info(f"Dataset saved successfully at {output_path}")


def main(
    num_laps_per_rider: int = 100,
    output_dir: str = "data/processed",
    seed: int = 42
):
    """
    Main pipeline for synthetic data generation.
    
    Args:
        num_laps_per_rider: Number of laps to generate per rider profile
        output_dir: Directory to save datasets
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    logger.info("Starting Minari dataset generation pipeline...")
    logger.info(f"Configuration: {num_laps_per_rider} laps per rider, seed={seed}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Generate Pro Rider data
        logger.info("\n--- Generating Pro Rider Data ---")
        pro_env = MotorcycleEnvWrapper(lap_length=300, seed=seed)
        pro_rider = ProRiderModel(seed=seed)
        pro_episodes = generate_minari_episodes(
            pro_env,
            pro_rider,
            num_laps=num_laps_per_rider,
            seed=seed,
            rider_name="pro"
        )
        
        pro_output_path = output_dir / "pro_rider_dataset.hdf5"
        save_to_minari_hdf5(pro_episodes, str(pro_output_path), rider_name="pro")
        
        # Generate Amateur Rider data
        logger.info("\n--- Generating Amateur Rider Data ---")
        amateur_env = MotorcycleEnvWrapper(lap_length=300, seed=seed + 1)
        amateur_rider = AmateurRiderModel(seed=seed + 1)
        amateur_episodes = generate_minari_episodes(
            amateur_env,
            amateur_rider,
            num_laps=num_laps_per_rider,
            seed=seed + 1,
            rider_name="amateur"
        )
        
        amateur_output_path = output_dir / "amateur_rider_dataset.hdf5"
        save_to_minari_hdf5(amateur_episodes, str(amateur_output_path), rider_name="amateur")
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("Dataset Generation Complete!")
        logger.info(f"Pro Rider Dataset: {pro_output_path}")
        logger.info(f"Amateur Rider Dataset: {amateur_output_path}")
        logger.info(f"Total episodes: {len(pro_episodes) + len(amateur_episodes)}")
        logger.info("="*60)
        
        return {
            'pro_dataset': str(pro_output_path),
            'amateur_dataset': str(amateur_output_path),
            'total_episodes': len(pro_episodes) + len(amateur_episodes)
        }
        
    except Exception as e:
        logger.error(f"Fatal error during data generation: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic motorcycle racing data in Minari format")
    parser.add_argument('--laps', type=int, default=100, help='Number of laps per rider')
    parser.add_argument('--output-dir', type=str, default='data/processed', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    main(
        num_laps_per_rider=args.laps,
        output_dir=args.output_dir,
        seed=args.seed
    )
