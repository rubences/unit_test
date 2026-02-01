"""
Stress-Aware Coaching Agent: PettingZoo environment with stress-based feedback blocking.

Logic:
- If pilot stress > 90%, haptic feedback is disabled to prevent panic
- Otherwise, feedback intensity is modulated by stress (lower stress = clearer feedback)
- Maintains training stability while ensuring pilot safety

This agent bridges the multimodal fusion network with the RL training pipeline.
"""

import logging
from typing import Dict, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class StressAwareCoachAgent:
    """Coaching agent with stress-aware haptic feedback control.

    Core Logic:
    - Monitors pilot stress level (from biometric fusion)
    - Blocks haptic feedback if stress > STRESS_THRESHOLD (default 0.9)
    - Modulates feedback intensity based on stress
    """

    STRESS_THRESHOLD = 0.90  # 90% stress threshold
    MIN_STRESS_FOR_FEEDBACK = 0.10  # Don't provide feedback below this

    def __init__(
        self,
        stress_threshold: float = 0.90,
        feedback_scaling: str = "linear",
    ):
        """Initialize stress-aware coach agent.

        Args:
            stress_threshold: Stress level [0, 1] above which feedback blocks
            feedback_scaling: How to scale feedback with stress ("linear", "exponential")
        """
        self.stress_threshold = stress_threshold
        self.feedback_scaling = feedback_scaling
        self.episode_stresses = []
        self.episode_blocked_steps = 0
        self.total_episodes = 0

    def compute_feedback_modulation(self, stress_level: float) -> float:
        """Compute feedback intensity modulation based on stress.

        Args:
            stress_level: Stress [0, 1]

        Returns:
            Modulation factor [0, 1] to apply to haptic feedback
        """
        if stress_level > self.stress_threshold:
            return 0.0  # Complete block

        if stress_level < self.MIN_STRESS_FOR_FEEDBACK:
            return 1.0  # Full feedback

        # Smooth transition in [MIN_STRESS, STRESS_THRESHOLD]
        normalized = (stress_level - self.MIN_STRESS_FOR_FEEDBACK) / (
            self.stress_threshold - self.MIN_STRESS_FOR_FEEDBACK
        )

        if self.feedback_scaling == "linear":
            return 1.0 - 0.5 * normalized  # Linear decay from 1.0 to 0.5
        elif self.feedback_scaling == "exponential":
            return np.exp(-2.0 * normalized)  # Exponential decay
        else:
            return 1.0 - 0.5 * normalized

    def apply_stress_blocking(
        self,
        haptic_action: np.ndarray,
        stress_level: float,
    ) -> Tuple[np.ndarray, Dict]:
        """Apply stress-based feedback blocking logic.

        Args:
            haptic_action: (3,) haptic action [left_intensity, right_intensity, frequency]
            stress_level: Current pilot stress [0, 1]

        Returns:
            (modified_action, metadata_dict)
        """
        metadata = {
            "stress_level": float(stress_level),
            "feedback_blocked": False,
            "modulation_factor": 1.0,
            "original_intensity": float(np.mean(haptic_action[:2])),
        }

        if stress_level > self.stress_threshold:
            # Complete feedback block
            modified_action = np.array([0.0, 0.0, haptic_action[2]], dtype=np.float32)
            metadata["feedback_blocked"] = True
            metadata["modulation_factor"] = 0.0
            self.episode_blocked_steps += 1

            logger.warning(
                f"⚠️  Stress block triggered (stress={stress_level:.2%}). Haptic feedback disabled."
            )
            return modified_action, metadata

        # Modulate feedback by stress level
        modulation = self.compute_feedback_modulation(stress_level)
        modified_action = haptic_action.copy()
        modified_action[0] *= modulation  # Left glove
        modified_action[1] *= modulation  # Right glove
        metadata["modulation_factor"] = modulation

        if modulation < 1.0:
            logger.info(
                f"ℹ️  Feedback modulated to {modulation:.1%} (stress={stress_level:.2%})"
            )

        return modified_action.astype(np.float32), metadata

    def reset_episode(self) -> None:
        """Reset episode counters."""
        self.total_episodes += 1
        if len(self.episode_stresses) > 0:
            avg_stress = np.mean(self.episode_stresses)
            pct_blocked = (self.episode_blocked_steps / max(len(self.episode_stresses), 1)) * 100
            logger.info(
                f"Episode {self.total_episodes}: Avg stress={avg_stress:.2%}, "
                f"Blocked steps={pct_blocked:.1f}%"
            )

        self.episode_stresses = []
        self.episode_blocked_steps = 0

    def log_step(self, stress_level: float) -> None:
        """Log stress for current step."""
        self.episode_stresses.append(stress_level)

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if len(self.episode_stresses) == 0:
            return {"episodes": 0, "avg_stress": 0.0, "peak_stress": 0.0, "blocks": 0}

        return {
            "episodes": self.total_episodes,
            "avg_stress": float(np.mean(self.episode_stresses)),
            "peak_stress": float(np.max(self.episode_stresses)),
            "blocks": self.episode_blocked_steps,
            "block_rate": float(self.episode_blocked_steps / len(self.episode_stresses)),
        }


class MultimodalRacingEnv(gym.Env):
    """Gymnasium environment with multimodal (telemetry + biometric) input.

    Observation space: (telemetry_seq, biometric_seq, stress_level)
    Action space: (left_intensity, right_intensity, frequency)
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        render_mode=None,
        seq_len: int = 128,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.seq_len = seq_len

        self.telemetry_history = np.zeros((seq_len, 6), dtype=np.float32)
        self.biometric_history = np.zeros((seq_len, 3), dtype=np.float32)
        self.current_stress = 0.0
        self.timestep = 0
        self.max_timesteps = 1000

        self.coach = StressAwareCoachAgent(stress_threshold=0.90)

        # Action space: [left_intensity, right_intensity, frequency]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 50.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 300.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation space: telemetry + biometrics + stress
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(seq_len * 6 + seq_len * 3 + 1,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)
        self.telemetry_history = np.zeros((self.seq_len, 6), dtype=np.float32)
        self.biometric_history = np.zeros((self.seq_len, 3), dtype=np.float32)
        self.current_stress = 0.0
        self.timestep = 0
        self.coach.reset_episode()
        return self._get_observation(), {}

    def step(self, action: np.ndarray):
        """Execute one step with stress-aware feedback blocking."""
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Update simulated biometrics (synthetic for demo)
        self._update_biometrics()

        # Apply stress blocking
        modulated_action, metadata = self.coach.apply_stress_blocking(action, self.current_stress)

        # Log stress for this step
        self.coach.log_step(self.current_stress)

        # Compute reward (simplified)
        reward = self._compute_reward(metadata)

        self.timestep += 1
        terminated = False  # Or use real termination logic
        truncated = self.timestep >= self.max_timesteps

        info = {
            "stress_level": self.current_stress,
            "feedback_blocked": metadata["feedback_blocked"],
            "modulation_factor": metadata["modulation_factor"],
        }

        return self._get_observation(), float(reward), terminated, truncated, info

    def _update_biometrics(self) -> None:
        """Simulate biometric evolution (HR, HRV, stress)."""
        # Synthetic brake signal simulation
        brake = 0.5 * (1 + np.sin(0.01 * self.timestep))

        # HR increases with braking
        hr = 70.0 + brake * 50.0 + 10 * np.random.randn()

        # HRV decreases with stress (inverse relationship)
        hrv = 50.0 * (1 - brake) + 5 * np.random.randn()

        # Stress = 1 - (HRV / HRV_max)
        self.current_stress = np.clip(1.0 - (hrv / 50.0), 0.0, 1.0)

        # Update history
        self.biometric_history = np.roll(self.biometric_history, -1, axis=0)
        self.biometric_history[-1] = np.array([hr, hrv, self.current_stress])

    def _update_telemetry(self) -> None:
        """Simulate telemetry (IMU signals)."""
        t = self.timestep / 50.0
        ax = 2.0 * np.sin(0.02 * t)
        ay = 1.5 * np.cos(0.015 * t)
        az = 9.81

        gx = 0.1 * np.sin(0.02 * t)
        gy = 0.2 * np.sin(0.025 * t)
        gz = 0.0

        self.telemetry_history = np.roll(self.telemetry_history, -1, axis=0)
        self.telemetry_history[-1] = np.array([ax, ay, az, gx, gy, gz])

    def _get_observation(self) -> np.ndarray:
        """Get flattened observation."""
        obs = np.concatenate([
            self.telemetry_history.flatten(),
            self.biometric_history.flatten(),
            np.array([self.current_stress]),
        ]).astype(np.float32)
        return obs

    def _compute_reward(self, metadata: Dict) -> float:
        """Compute reward with stress considerations."""
        reward = 0.0

        # Reward for low stress
        stress_penalty = 0.5 * self.current_stress**2
        reward -= stress_penalty

        # Penalize feedback blocks
        if metadata["feedback_blocked"]:
            reward -= 1.0

        return reward

    def render(self):
        """Render environment (placeholder)."""
        if self.render_mode == "human":
            print(f"Step {self.timestep}: stress={self.current_stress:.2%}")


if __name__ == "__main__":
    # Demo: Test stress-aware agent
    print("=" * 60)
    print("Stress-Aware Coaching Agent Demo")
    print("=" * 60)

    agent = StressAwareCoachAgent(stress_threshold=0.90)

    # Test cases
    test_stresses = [0.0, 0.5, 0.85, 0.90, 0.95, 1.0]
    haptic_action = np.array([1.0, 1.0, 200.0])

    print("\nStress Blocking Logic:")
    for stress in test_stresses:
        modulated, metadata = agent.apply_stress_blocking(haptic_action, stress)
        status = "BLOCKED" if metadata["feedback_blocked"] else "ACTIVE"
        print(
            f"  Stress {stress:.0%}: {status:7} | "
            f"Modulation: {metadata['modulation_factor']:.1%} | "
            f"Left intensity: {modulated[0]:.2f}"
        )

    print("\n✓ Stress-aware agent demo complete")
