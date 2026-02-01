"""
Adversarial Sensor Noise Agent: PettingZoo agent that attacks telemetry data.

Role: "Villano" que inyecta ruido, drift y cortes de señal en los sensores IMU.

Estrategias de Ataque:
1. Gaussian Noise: Ruido blanco N(0, σ²) en sensores específicos
2. Sensor Drift: Desviación sistemática que aumenta con el tiempo
3. Signal Cutout: Pone a cero ciertos sensores (intermitente)
4. Bias Injection: Agrega sesgo constante a lecturas

Curriculum Learning:
- Etapa 1 (Easy): Ruido débil, baja probabilidad de cutout
- Etapa 2 (Medium): Ruido moderado, cutout más frecuente
- Etapa 3 (Hard): Ruido fuerte, drift significativo, cutout agresivo
"""

import logging
from typing import Dict, Tuple, Any, Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class SensorNoiseAgent:
    """Adversarial agent that attacks IMU telemetry with realistic sensor faults.
    
    Attack Modes:
    - Gaussian Noise: σ = noise_level * sensor_range
    - Drift: Accelerated bias that increases with time (δ per step)
    - Cutout: Probabilistic sensor dropout (p_cutout)
    - Bias: Constant offset on specific axes
    """

    def __init__(
        self,
        noise_level: float = 0.0,
        curriculum_stage: int = 1,
        attack_modes: Optional[list] = None,
        seed: int = 42,
    ):
        """Initialize adversarial sensor noise agent.

        Args:
            noise_level: Base noise intensity [0, 1] where 1.0 = 100% of sensor range
            curriculum_stage: Training stage (1=easy, 2=medium, 3=hard)
            attack_modes: List of attack modes to enable (default: all)
            seed: Random seed
        """
        self.noise_level = noise_level
        self.curriculum_stage = curriculum_stage
        self.attack_modes = attack_modes or ["gaussian", "drift", "cutout", "bias"]
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Curriculum progression
        self.stage_params = self._get_stage_params(curriculum_stage)
        
        # Tracking for drift
        self.drift_accumulators = {}  # {axis: accumulated_drift}
        self.step_count = 0
        
        logger.info(
            f"SensorNoiseAgent initialized: noise={noise_level:.3f}, "
            f"stage={curriculum_stage}, modes={self.attack_modes}"
        )

    def _get_stage_params(self, stage: int) -> Dict[str, float]:
        """Get curriculum parameters for given stage.

        Args:
            stage: Curriculum stage (1=easy, 2=medium, 3=hard)

        Returns:
            Dict with attack parameters
        """
        params = {
            1: {  # Easy: Weak attacks
                "gaussian_scale": 0.1,
                "drift_rate": 0.001,
                "cutout_prob": 0.05,
                "bias_magnitude": 0.05,
            },
            2: {  # Medium: Moderate attacks
                "gaussian_scale": 0.3,
                "drift_rate": 0.005,
                "cutout_prob": 0.15,
                "bias_magnitude": 0.15,
            },
            3: {  # Hard: Aggressive attacks
                "gaussian_scale": 0.5,
                "drift_rate": 0.01,
                "cutout_prob": 0.30,
                "bias_magnitude": 0.30,
            },
        }
        return params.get(stage, params[1])

    def set_curriculum_stage(self, stage: int) -> None:
        """Update curriculum stage and adjust attack intensity.

        Args:
            stage: New curriculum stage (1, 2, or 3)
        """
        self.curriculum_stage = max(1, min(3, stage))
        self.stage_params = self._get_stage_params(self.curriculum_stage)
        logger.info(f"Curriculum advanced to stage {self.curriculum_stage}")

    def inject_noise(self, telemetry: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Inject noise into telemetry data using curriculum-controlled attacks.

        Telemetry format (6-dim IMU):
        [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]

        Args:
            telemetry: (6,) array of IMU telemetry

        Returns:
            (corrupted_telemetry, metadata_dict)
        """
        corrupted = telemetry.copy().astype(np.float32)
        metadata = {
            "noise_level": self.noise_level,
            "curriculum_stage": self.curriculum_stage,
            "attacks_applied": [],
            "perturbation_magnitude": 0.0,
        }

        # Estimate sensor ranges (typical IMU ranges)
        accel_range = 16.0  # ±16 G for accelerometer
        gyro_range = 2000.0  # ±2000 °/s for gyroscope
        sensor_ranges = np.array(
            [accel_range, accel_range, accel_range, gyro_range, gyro_range, gyro_range]
        )

        # Initialize drift accumulators
        for i in range(len(corrupted)):
            if i not in self.drift_accumulators:
                self.drift_accumulators[i] = 0.0

        perturbation = np.zeros_like(corrupted)

        # 1. Gaussian Noise
        if "gaussian" in self.attack_modes:
            sigma = (
                self.noise_level
                * self.stage_params["gaussian_scale"]
                * sensor_ranges
            )
            gaussian_noise = self.rng.normal(0, sigma)
            corrupted += gaussian_noise
            perturbation += gaussian_noise
            metadata["attacks_applied"].append("gaussian")

        # 2. Drift (accelerating bias)
        if "drift" in self.attack_modes:
            drift_rate = self.noise_level * self.stage_params["drift_rate"]
            for i in range(len(corrupted)):
                self.drift_accumulators[i] += drift_rate * sensor_ranges[i]
                corrupted[i] += self.drift_accumulators[i]
                perturbation[i] += self.drift_accumulators[i]
            metadata["attacks_applied"].append("drift")

        # 3. Signal Cutout (intermittent sensor failure)
        if "cutout" in self.attack_modes:
            cutout_prob = self.noise_level * self.stage_params["cutout_prob"]
            cutout_mask = self.rng.rand(len(corrupted)) < cutout_prob
            corrupted[cutout_mask] = 0.0
            perturbation[cutout_mask] = 0.0
            if cutout_mask.any():
                metadata["attacks_applied"].append(f"cutout({cutout_mask.sum()})")

        # 4. Bias Injection (constant offset)
        if "bias" in self.attack_modes:
            bias_mag = self.noise_level * self.stage_params["bias_magnitude"] * sensor_ranges
            bias = self.rng.uniform(-bias_mag, bias_mag)
            corrupted += bias
            perturbation += bias
            metadata["attacks_applied"].append("bias")

        metadata["perturbation_magnitude"] = float(np.linalg.norm(perturbation))
        self.step_count += 1

        return corrupted, metadata

    def reset_drift(self) -> None:
        """Reset drift accumulators and step counter (called at episode start)."""
        self.drift_accumulators = {}
        self.step_count = 0

    def get_attack_strength(self) -> float:
        """Get current attack strength as scalar [0, 1].
        
        Returns:
            Effective attack strength
        """
        stage_factor = self.curriculum_stage / 3.0
        return self.noise_level * stage_factor

    def get_status(self) -> Dict[str, Any]:
        """Get detailed status of attacker.

        Returns:
            Status dictionary with current configuration
        """
        return {
            "noise_level": self.noise_level,
            "curriculum_stage": self.curriculum_stage,
            "attack_modes": self.attack_modes,
            "step_count": self.step_count,
            "stage_params": self.stage_params,
            "attack_strength": self.get_attack_strength(),
        }


class AdversarialEnvironmentWrapper:
    """Gymnasium wrapper that adds adversarial sensor attacks to any environment.

    This wrapper:
    1. Extracts IMU telemetry from observation
    2. Passes it through SensorNoiseAgent
    3. Reinjected corrupted telemetry to observation
    """

    def __init__(
        self,
        env,
        sensor_noise_agent: Optional[SensorNoiseAgent] = None,
        telemetry_indices: Tuple[int, ...] = (0, 1, 2, 3, 4, 5),
    ):
        """Initialize adversarial wrapper.

        Args:
            env: Gymnasium environment to wrap
            sensor_noise_agent: SensorNoiseAgent instance (created if None)
            telemetry_indices: Indices in observation corresponding to IMU sensors
        """
        self.env = env
        self.sensor_noise_agent = sensor_noise_agent or SensorNoiseAgent()
        self.telemetry_indices = telemetry_indices
        self.episode_attacks = []

    def reset(self, seed=None, options=None):
        """Reset environment and attack tracking."""
        obs, info = self.env.reset(seed=seed, options=options)
        self.sensor_noise_agent.reset_drift()
        self.episode_attacks = []
        return obs, info

    def step(self, action):
        """Step environment and inject adversarial noise."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Extract telemetry
        obs_list = list(obs) if isinstance(obs, (tuple, list, np.ndarray)) else [obs]
        telemetry = np.array([obs_list[i] for i in self.telemetry_indices])

        # Inject noise
        corrupted_telemetry, attack_info = self.sensor_noise_agent.inject_noise(telemetry)
        self.episode_attacks.append(attack_info)

        # Reinjected corrupted telemetry
        for i, idx in enumerate(self.telemetry_indices):
            if idx < len(obs_list):
                obs_list[idx] = corrupted_telemetry[i]

        obs = np.array(obs_list)
        info["adversarial"] = attack_info

        return obs, reward, terminated, truncated, info

    def set_noise_level(self, noise_level: float) -> None:
        """Update noise level (for curriculum learning)."""
        self.sensor_noise_agent.noise_level = noise_level

    def set_curriculum_stage(self, stage: int) -> None:
        """Update curriculum stage."""
        self.sensor_noise_agent.set_curriculum_stage(stage)

    def get_episode_stats(self) -> Dict[str, Any]:
        """Get statistics about attacks applied in episode."""
        if not self.episode_attacks:
            return {"total_attacks": 0}

        perturbations = [a.get("perturbation_magnitude", 0) for a in self.episode_attacks]
        return {
            "total_attacks": len(self.episode_attacks),
            "avg_perturbation": float(np.mean(perturbations)),
            "max_perturbation": float(np.max(perturbations)),
            "min_perturbation": float(np.min(perturbations)),
        }

    def close(self):
        """Close wrapped environment."""
        if hasattr(self.env, "close"):
            self.env.close()
