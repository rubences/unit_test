"""
Biometric-Aware Motorcycle Environment: Gymnasium wrapper with ECG integration

Environment Features:
- Extended observation space: [speed, lean_angle, g_force, hr_normalized, rmssd_index]
- Panic Freeze safety rule: If RMSSD collapses + high G-force → force haptic_intensity=0
- Physiological constraints on action execution
- Real-time stress monitoring

Safety Implementation:
- Monitors pilot's cognitive load via RMSSD
- Triggers "Panic Freeze" when:
  1. RMSSD drops below 10ms (cognitive saturation)
  2. AND G-force > 1.2G (high physical stress)
  3. → Force haptic_intensity = 0 to prevent overload
"""

import gymnasium as gym
import numpy as np
import logging
from typing import Tuple, Dict, Optional, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MotorcycleBioEnv(gym.Env):
    """
    Motorcycle environment with biometric feedback integration.
    
    State Space:
    [0] speed: m/s (0-80)
    [1] lean_angle: degrees (0-65)
    [2] g_force: G (0-2)
    [3] hr_normalized: (0-1) normalized HR
    [4] rmssd_index: (0-1) HRV indicator
    
    Action Space:
    [0] throttle: (0-1)
    [1] brake: (0-1)
    [2] lean_input: (-1 to 1)
    [3] haptic_intensity: (0-1)
    
    Observation Space:
    Box(5,) - state + biometric features
    """
    
    metadata = {'render_modes': ['human'], 'render_fps': 30}
    
    def __init__(
        self,
        max_episode_steps: int = 1000,
        render_mode: Optional[str] = None,
        panic_freeze_enabled: bool = True,
    ):
        """
        Initialize biometric-aware motorcycle environment.
        
        Args:
            max_episode_steps: Maximum steps per episode
            render_mode: Rendering mode ('human' or None)
            panic_freeze_enabled: Enable panic safety mechanism
        """
        super().__init__()
        
        # Configuration
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.panic_freeze_enabled = panic_freeze_enabled
        
        # Physical parameters
        self.max_speed = 80.0  # m/s
        self.max_lean_angle = 65.0  # degrees
        self.max_g_force = 2.0  # G
        
        # Safety thresholds
        self.rmssd_panic_threshold = 10.0  # ms (cognitive saturation)
        self.g_force_danger_threshold = 1.2  # G
        self.hr_warning_threshold = 170  # bpm
        
        # State variables
        self.state = None
        self.step_count = 0
        self.episode_count = 0
        
        # Biometric state
        self.hr = 110.0
        self.rmssd = 40.0
        self.stress_index = 0.0
        
        # Panic freeze tracking
        self.panic_frozen = False
        self.panic_freeze_count = 0
        
        # Action/Observation spaces
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([self.max_speed, self.max_lean_angle, self.max_g_force, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        
        logger.info("✓ MotorcycleBioEnv initialized (panic_freeze=%s)" % panic_freeze_enabled)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options
        
        Returns:
            observation, info
        """
        super().reset(seed=seed)
        
        # Initial state
        self.state = np.array([
            10.0,    # speed (starting slow)
            0.0,     # lean_angle
            0.0,     # g_force
            0.5,     # hr_normalized (baseline)
            0.8,     # rmssd_index (relaxed)
        ], dtype=np.float32)
        
        # Biometric reset
        self.hr = 110.0
        self.rmssd = 40.0
        self.stress_index = 0.0
        
        # Reset counters
        self.step_count = 0
        self.panic_frozen = False
        self.panic_freeze_count = 0
        self.episode_count += 1
        
        info = {
            'episode': self.episode_count,
            'hr': self.hr,
            'rmssd': self.rmssd,
            'stress': self.stress_index,
            'panic_frozen': False,
        }
        
        return self.state, info
    
    def _update_biometrics(self, g_force: float, speed: float, lean_angle: float) -> None:
        """
        Update biometric state based on physical stress.
        
        Simulation Logic:
        - HR increases with G-force and speed
        - RMSSD collapses under sustained stress
        - Stress index integrates both metrics
        
        Args:
            g_force: Current G-force
            speed: Current speed
            lean_angle: Current lean angle
        """
        # Compute stress level from physics
        stress_physics = (
            0.4 * min(g_force / 2.0, 1.0) +  # G-force component
            0.3 * min(speed / 80.0, 1.0) +   # Speed component
            0.3 * min(lean_angle / 65.0, 1.0)  # Lean angle component
        )
        
        # Update HR (smooth exponential)
        hr_baseline = 110.0
        hr_peak = 180.0
        target_hr = hr_baseline + (hr_peak - hr_baseline) * stress_physics
        self.hr = 0.8 * self.hr + 0.2 * target_hr  # Low-pass filter
        
        # Update RMSSD (exponential decay with stress)
        rmssd_baseline = 60.0
        rmssd_panic = 8.0
        target_rmssd = rmssd_baseline * np.exp(-2.0 * stress_physics)
        target_rmssd = np.clip(target_rmssd, rmssd_panic, rmssd_baseline)
        self.rmssd = 0.8 * self.rmssd + 0.2 * target_rmssd
        
        # Update stress index
        self.stress_index = stress_physics
    
    def _check_panic_freeze(self, g_force: float) -> bool:
        """
        Check if panic freeze should be activated.
        
        Panic Freeze Rule:
        IF rmssd < 10ms (cognitive saturation) AND g_force > 1.2G
        THEN force haptic_intensity = 0 for safety
        
        Args:
            g_force: Current G-force
        
        Returns:
            True if panic freeze is active
        """
        if not self.panic_freeze_enabled:
            return False
        
        # Check conditions
        cognitive_saturation = self.rmssd < self.rmssd_panic_threshold
        high_physical_stress = g_force > self.g_force_danger_threshold
        
        if cognitive_saturation and high_physical_stress:
            if not self.panic_frozen:
                logger.warning(f"⚠ PANIC FREEZE activated! RMSSD={self.rmssd:.1f}ms, G={g_force:.2f}G")
                self.panic_frozen = True
            self.panic_freeze_count += 1
            return True
        else:
            self.panic_frozen = False
            return False
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Panic Freeze Application:
        1. Process action normally
        2. Check panic conditions
        3. If panic → force haptic_intensity = 0, ignore coaching
        
        Args:
            action: [throttle, brake, lean_input, haptic_intensity]
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Unpack action
        throttle = np.clip(action[0], 0, 1)
        brake = np.clip(action[1], 0, 1)
        lean_input = np.clip(action[2], -1, 1)
        haptic_intensity = np.clip(action[3], 0, 1)
        
        # Current state
        speed, lean_angle, g_force, hr_norm, rmssd_idx = self.state
        
        # Physics simulation (simplified)
        # Speed update
        acceleration = 2.0 * throttle - 3.0 * brake
        speed = np.clip(speed + acceleration * 0.1, 0, self.max_speed)
        
        # Lean angle update
        lean_angle = np.clip(
            lean_angle + lean_input * 5.0,
            0,
            self.max_lean_angle
        )
        
        # G-force computation (simplified circular motion)
        # g_force ≈ speed^2 / radius, where radius depends on lean
        # Lean angle → turning radius
        if lean_angle > 0:
            radius = 50.0 / (1.0 + lean_angle / 30.0)  # Tighter lean = smaller radius
            g_lateral = (speed ** 2) / (radius * 9.81) / 10.0
        else:
            g_lateral = 0.0
        
        g_brake = 3.0 * brake  # Braking G-force
        g_force = np.sqrt(g_lateral ** 2 + g_brake ** 2)
        g_force = np.clip(g_force, 0, self.max_g_force)
        
        # Update biometrics
        self._update_biometrics(g_force, speed, lean_angle)
        
        # Check panic freeze
        is_panic_frozen = self._check_panic_freeze(g_force)
        
        if is_panic_frozen and self.panic_freeze_enabled:
            # Force haptic intensity to 0 (safety override)
            haptic_intensity = 0.0
            action = np.array([throttle, brake, lean_input, 0.0], dtype=np.float32)
        
        # Normalize biometric features for observation
        hr_normalized = np.clip((self.hr - 50) / 150, 0, 1)
        rmssd_normalized = np.clip((self.rmssd - 8) / 52, 0, 1)
        
        # Update state
        self.state = np.array([
            speed,
            lean_angle,
            g_force,
            hr_normalized,
            rmssd_normalized,
        ], dtype=np.float32)
        
        # Compute reward
        # Reward for smooth driving (low stress), penalty for panic
        reward = 0.0
        
        # Positive reward for efficient cornering
        if lean_angle > 10:
            reward += (1.0 - self.stress_index) * speed / 80.0
        else:
            reward += speed / 100.0  # Reward speed on straights
        
        # Penalty for high stress
        if self.stress_index > 0.7:
            reward -= 0.5
        
        # Penalty for panic freeze
        if is_panic_frozen:
            reward -= 0.3
        
        # Termination conditions
        done = (
            speed <= 0.0 or
            self.step_count >= self.max_episode_steps or
            self.panic_freeze_count > 50  # Too many panic events
        )
        
        self.step_count += 1
        
        # Info
        info = {
            'step': self.step_count,
            'episode': self.episode_count,
            'hr': self.hr,
            'rmssd': self.rmssd,
            'stress': self.stress_index,
            'panic_frozen': is_panic_frozen,
            'panic_freeze_count': self.panic_freeze_count,
            'haptic_intensity_applied': float(haptic_intensity),
        }
        
        return self.state, reward, done, False, info
    
    def set_biometric_state(self, hr: float, rmssd: float) -> None:
        """
        Directly set biometric state (for testing/simulation).
        
        Args:
            hr: Heart rate (bpm)
            rmssd: RMSSD (ms)
        """
        self.hr = np.clip(hr, 40, 200)
        self.rmssd = np.clip(rmssd, 5, 100)
        
        # Update normalized values in state
        hr_normalized = np.clip((self.hr - 50) / 150, 0, 1)
        rmssd_normalized = np.clip((self.rmssd - 8) / 52, 0, 1)
        
        self.state[3] = hr_normalized
        self.state[4] = rmssd_normalized
    
    def get_danger_zone_info(self) -> Dict[str, bool]:
        """
        Return information about danger zones.
        
        Returns:
            dict with:
                - 'cognitive_saturation': RMSSD < panic threshold
                - 'high_g_force': G-force > danger threshold
                - 'panic_active': Both conditions met
        """
        return {
            'cognitive_saturation': self.rmssd < self.rmssd_panic_threshold,
            'high_g_force': self.state[2] > self.g_force_danger_threshold,
            'panic_active': self.panic_frozen,
            'rmssd': self.rmssd,
            'g_force': self.state[2],
        }
    
    def render(self) -> None:
        """Render environment state (placeholder)."""
        if self.render_mode == 'human':
            print(f"[Step {self.step_count}] "
                  f"Speed: {self.state[0]:.1f} m/s, "
                  f"Lean: {self.state[1]:.1f}°, "
                  f"G: {self.state[2]:.2f}G, "
                  f"HR: {self.hr:.0f} bpm, "
                  f"RMSSD: {self.rmssd:.1f} ms, "
                  f"Panic: {self.panic_frozen}")


# Register environment
def register_env():
    """Register custom environment with gymnasium."""
    try:
        gym.register(
            id='MotorcycleBio-v0',
            entry_point='src.environments.moto_bio_env:MotorcycleBioEnv',
            max_episode_steps=1000,
            kwargs={'panic_freeze_enabled': True},
        )
        logger.info("✓ Environment registered as 'MotorcycleBio-v0'")
    except Exception as e:
        logger.warning(f"Environment already registered or error: {e}")


if __name__ == '__main__':
    # Demo
    logger.info("=" * 60)
    logger.info("MotorcycleBioEnv Demo")
    logger.info("=" * 60)
    
    # Create environment
    env = MotorcycleBioEnv(panic_freeze_enabled=True)
    
    # Episode
    logger.info("\n1. Running episode...")
    obs, info = env.reset()
    
    for step in range(50):
        # Random action
        action = env.action_space.sample()
        
        obs, reward, done, truncated, info = env.step(action)
        
        if step % 10 == 0:
            logger.info(f"   Step {step}: HR={info['hr']:.0f} bpm, "
                       f"RMSSD={info['rmssd']:.1f} ms, "
                       f"Panic={info['panic_frozen']}")
        
        if done:
            break
    
    # Danger zone test
    logger.info("\n2. Testing panic freeze...")
    env.reset()
    env.set_biometric_state(hr=170, rmssd=9.0)  # Cognitive saturation
    
    action = np.array([0.5, 0.0, 0.8, 1.0], dtype=np.float32)  # High lean + haptic
    obs, reward, done, truncated, info = env.step(action)
    
    danger = env.get_danger_zone_info()
    logger.info(f"   Danger zone: {danger}")
    logger.info(f"   Haptic applied: {info['haptic_intensity_applied']}")
    
    logger.info("\n✓ Demo complete")
