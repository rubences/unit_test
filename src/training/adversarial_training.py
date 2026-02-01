"""
Adversarial Training Script: Curriculum Learning for Robust Motorcycle Coaching.

Pipeline:
1. Train baseline model (SB3 agent) without adversarial noise
2. Train adversarial model with curriculum learning (stages 1→2→3)
3. Compare robustness across noise levels

Curriculum Schedule:
- Epochs 1-3: Stage 1 (Easy) - σ=0.1, p_cutout=5%, drift_rate=0.001
- Epochs 4-6: Stage 2 (Medium) - σ=0.3, p_cutout=15%, drift_rate=0.005
- Epochs 7-10: Stage 3 (Hard) - σ=0.5, p_cutout=30%, drift_rate=0.01

Evaluation:
- Test on noise levels [0%, 5%, 10%, 15%, 20%, 25%]
- Metrics: Success rate, prediction error, trajectory deviation
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import json
from dataclasses import dataclass, asdict

try:
    from stable_baselines3 import PPO, A2C, DQN
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for adversarial training."""
    
    # Training
    total_timesteps: int = 100_000
    n_envs: int = 4
    algo: str = "PPO"  # PPO, A2C, or DQN
    
    # Curriculum learning
    curriculum_enabled: bool = True
    stage_duration: int = 10_000  # timesteps per stage
    
    # Adversarial attack
    max_noise_level: float = 0.20  # 20% sensor noise at full stage 3
    attack_modes: List[str] = None
    
    # Evaluation
    eval_noise_levels: List[float] = None
    eval_episodes: int = 10
    
    # Checkpointing
    save_dir: str = "models/adversarial"
    save_freq: int = 10_000
    
    def __post_init__(self):
        if self.attack_modes is None:
            self.attack_modes = ["gaussian", "drift", "cutout", "bias"]
        if self.eval_noise_levels is None:
            self.eval_noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]


class CurriculumCallback(BaseCallback):
    """Callback that advances curriculum during training."""

    def __init__(
        self,
        env_wrapper,
        curriculum_config: Dict[str, int],
        verbose: int = 1,
    ):
        """Initialize curriculum callback.

        Args:
            env_wrapper: AdversarialEnvironmentWrapper instance
            curriculum_config: {"stage_duration": int, "max_stages": int}
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.env_wrapper = env_wrapper
        self.stage_duration = curriculum_config.get("stage_duration", 10_000)
        self.max_stages = curriculum_config.get("max_stages", 3)
        self.current_stage = 1

    def _on_step(self) -> bool:
        """Advance curriculum at end of each stage."""
        timestep = self.num_timesteps
        expected_stage = min(
            self.max_stages,
            (timestep // self.stage_duration) + 1
        )

        if expected_stage > self.current_stage:
            self.current_stage = expected_stage
            self.env_wrapper.set_curriculum_stage(self.current_stage)
            
            if self.verbose >= 1:
                logger.info(
                    f"Curriculum advanced to STAGE {self.current_stage} "
                    f"at timestep {timestep}"
                )

        return True


def create_dummy_env(noise_level: float = 0.0, curriculum_stage: int = 1):
    """Create a dummy environment for testing (without actual simulator).

    Args:
        noise_level: Sensor noise level [0, 1]
        curriculum_stage: Curriculum stage (1-3)

    Returns:
        Gymnasium environment
    """
    from gym import spaces
    
    class DummyMotorcycleEnv(gym.Env):
        """Minimal dummy environment for testing training pipeline."""
        
        def __init__(self):
            super().__init__()
            # Observation: [telemetry_6d, biometric_3d, state_3d] = 12d
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
            )
            # Action: [haptic_left, haptic_right, haptic_freq] = 3d
            self.action_space = spaces.Box(
                low=0.0, high=1.0, shape=(3,), dtype=np.float32
            )
            self.step_count = 0
            self.max_steps = 1000

        def reset(self, seed=None, options=None):
            self.step_count = 0
            obs = self.observation_space.sample()
            return obs, {}

        def step(self, action):
            self.step_count += 1
            
            # Dummy reward: encourage non-zero haptic feedback
            reward = float(np.sum(np.abs(action[:2])) * 0.1)
            
            # Generate random observation
            obs = self.observation_space.sample()
            
            terminated = self.step_count >= self.max_steps
            truncated = False
            
            return obs, reward, terminated, truncated, {}
    
    return DummyMotorcycleEnv()


def train_baseline(config: TrainingConfig) -> Tuple[Any, Dict]:
    """Train baseline model without adversarial noise.

    Args:
        config: TrainingConfig instance

    Returns:
        (trained_model, metrics_dict)
    """
    logger.info("="*60)
    logger.info("TRAINING BASELINE (NO ADVERSARIAL NOISE)")
    logger.info("="*60)

    # Create environment
    env = create_dummy_env(noise_level=0.0)

    # Create model
    if config.algo == "PPO":
        model = PPO("MlpPolicy", env, verbose=1, n_steps=2048)
    elif config.algo == "A2C":
        model = A2C("MlpPolicy", env, verbose=1)
    else:
        model = PPO("MlpPolicy", env, verbose=1)

    # Train
    model.learn(total_timesteps=config.total_timesteps)

    logger.info(f"Baseline training complete ({config.total_timesteps} timesteps)")

    return model, {
        "model_type": "baseline",
        "algo": config.algo,
        "timesteps": config.total_timesteps,
    }


def train_adversarial(config: TrainingConfig) -> Tuple[Any, Dict]:
    """Train model with adversarial curriculum learning.

    Args:
        config: TrainingConfig instance

    Returns:
        (trained_model, metrics_dict)
    """
    logger.info("="*60)
    logger.info("TRAINING ADVERSARIAL (CURRICULUM LEARNING)")
    logger.info("="*60)

    from src.agents.sensor_noise_agent import SensorNoiseAgent, AdversarialEnvironmentWrapper

    # Create base environment
    env = create_dummy_env(noise_level=0.0)

    # Wrap with adversarial noise
    noise_agent = SensorNoiseAgent(
        noise_level=0.0,  # Start weak, will increase with curriculum
        curriculum_stage=1,
        attack_modes=config.attack_modes,
    )
    adversarial_env = AdversarialEnvironmentWrapper(
        env, sensor_noise_agent=noise_agent
    )

    # Create model
    if config.algo == "PPO":
        model = PPO("MlpPolicy", adversarial_env, verbose=1, n_steps=2048)
    elif config.algo == "A2C":
        model = A2C("MlpPolicy", adversarial_env, verbose=1)
    else:
        model = PPO("MlpPolicy", adversarial_env, verbose=1)

    # Curriculum callback
    if config.curriculum_enabled:
        callback = CurriculumCallback(
            adversarial_env,
            {
                "stage_duration": config.stage_duration,
                "max_stages": 3,
            },
        )
    else:
        callback = None

    # Train with curriculum
    logger.info(f"Training with curriculum: {config.stage_duration} timesteps per stage")
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callback,
    )

    logger.info(f"Adversarial training complete ({config.total_timesteps} timesteps)")

    return model, {
        "model_type": "adversarial",
        "algo": config.algo,
        "timesteps": config.total_timesteps,
        "curriculum_enabled": config.curriculum_enabled,
        "max_noise_level": config.max_noise_level,
        "attack_modes": config.attack_modes,
    }


def evaluate_robustness(
    models: Dict[str, Any],
    config: TrainingConfig,
) -> Dict[str, Any]:
    """Evaluate model robustness across noise levels.

    Args:
        models: Dict mapping model names to trained models
        config: TrainingConfig instance

    Returns:
        Results dictionary
    """
    from src.agents.sensor_noise_agent import SensorNoiseAgent, AdversarialEnvironmentWrapper

    logger.info("="*60)
    logger.info("EVALUATING ROBUSTNESS")
    logger.info("="*60)

    results = {
        "noise_levels": config.eval_noise_levels,
        "models": {},
    }

    for model_name, model in models.items():
        logger.info(f"\nEvaluating {model_name}...")
        model_results = {
            "mean_rewards": [],
            "std_rewards": [],
            "success_rates": [],
            "perturbations": [],
        }

        for noise_level in config.eval_noise_levels:
            logger.info(f"  Noise level: {noise_level:.1%}")

            # Create test environment with noise
            env = create_dummy_env(noise_level=noise_level)
            noise_agent = SensorNoiseAgent(
                noise_level=noise_level,
                curriculum_stage=3,  # Use hard stage for evaluation
            )
            test_env = AdversarialEnvironmentWrapper(env, sensor_noise_agent=noise_agent)

            # Run evaluation episodes
            episode_rewards = []
            all_perturbations = []

            for ep in range(config.eval_episodes):
                obs, info = test_env.reset()
                ep_reward = 0.0
                done = False

                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = test_env.step(action)
                    ep_reward += reward
                    done = terminated or truncated

                    if "adversarial" in info:
                        pert = info["adversarial"].get("perturbation_magnitude", 0.0)
                        all_perturbations.append(pert)

                episode_rewards.append(ep_reward)

            # Compute statistics
            mean_reward = float(np.mean(episode_rewards))
            std_reward = float(np.std(episode_rewards))
            success_rate = float(np.sum(np.array(episode_rewards) > 0) / len(episode_rewards))
            mean_pert = float(np.mean(all_perturbations)) if all_perturbations else 0.0

            model_results["mean_rewards"].append(mean_reward)
            model_results["std_rewards"].append(std_reward)
            model_results["success_rates"].append(success_rate)
            model_results["perturbations"].append(mean_pert)

            logger.info(
                f"    Mean Reward: {mean_reward:.3f}±{std_reward:.3f}, "
                f"Success: {success_rate:.1%}"
            )

            test_env.close()

        results["models"][model_name] = model_results

    return results


def main():
    """Main training and evaluation pipeline."""
    
    logger.info("ADVERSARIAL TRAINING PIPELINE")
    logger.info("="*60)

    # Configuration
    config = TrainingConfig(
        total_timesteps=50_000,  # Reduced for demo
        stage_duration=10_000,
        max_noise_level=0.20,
        eval_episodes=5,  # Reduced for demo
    )

    logger.info(f"Config: {asdict(config)}")

    # Create save directory
    Path(config.save_dir).mkdir(parents=True, exist_ok=True)

    # Train models
    logger.info("\n" + "="*60)
    logger.info("PHASE 1: MODEL TRAINING")
    logger.info("="*60)

    baseline_model, baseline_info = train_baseline(config)
    adversarial_model, adversarial_info = train_adversarial(config)

    models = {
        "Baseline": baseline_model,
        "Adversarial": adversarial_model,
    }

    # Evaluate robustness
    logger.info("\n" + "="*60)
    logger.info("PHASE 2: ROBUSTNESS EVALUATION")
    logger.info("="*60)

    results = evaluate_robustness(models, config)

    # Save results
    results_path = Path(config.save_dir) / "robustness_results.json"
    with open(results_path, "w") as f:
        # Convert numpy types to native Python for JSON serialization
        results_serializable = {
            "noise_levels": [float(x) for x in results["noise_levels"]],
            "models": {
                name: {
                    k: [float(v) for v in vals] if isinstance(vals, list) else vals
                    for k, vals in data.items()
                }
                for name, data in results["models"].items()
            },
        }
        json.dump(results_serializable, f, indent=2)

    logger.info(f"\nResults saved to {results_path}")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("ROBUSTNESS SUMMARY")
    logger.info("="*60)

    for model_name, model_data in results["models"].items():
        logger.info(f"\n{model_name}:")
        for i, noise_level in enumerate(results["noise_levels"]):
            reward = model_data["mean_rewards"][i]
            success = model_data["success_rates"][i]
            logger.info(
                f"  {noise_level:5.0%} noise: "
                f"Reward={reward:7.3f}, Success={success:5.1%}"
            )

    return results


if __name__ == "__main__":
    results = main()
