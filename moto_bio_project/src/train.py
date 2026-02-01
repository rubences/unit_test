"""
PPO Training Module with Callbacks and Evaluation
Trains a policy on MotoBioEnv using Stable-Baselines3
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import warnings
import os

# Suprimir warnings de Gym y Gymnasium
warnings.filterwarnings('ignore')
os.environ['GYM_IGNORE_DEPRECATION_WARNING'] = '1'

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Soporte para importación relativa y absoluta
try:
    from .config import TRAIN_CONFIG, PATHS, SIM_CONFIG
    from .environment import MotoBioEnv
    from .data_gen import SyntheticTelemetry
except ImportError:
    from config import TRAIN_CONFIG, PATHS, SIM_CONFIG
    from environment import MotoBioEnv
    from data_gen import SyntheticTelemetry


class BioAdaptiveCallback(BaseCallback):
    """
    Custom callback for tracking bio-adaptive metrics during training
    
    Logs: episode rewards, bio-gate activations, off-track events
    """
    
    def __init__(self, verbose: int = 0):
        """Initialize callback"""
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.bio_gate_rates = []
        self.off_track_rates = []
        self.episode_count = 0
    
    def _on_step(self) -> bool:
        """Called at each step"""
        # Check if episode just finished
        if self.locals.get('dones')[0]:
            self.episode_count += 1
            
            # Extract episode info from info dict
            info = self.locals.get('infos')[0]
            episode_reward = self.locals.get('rewards')[0]
            
            self.episode_rewards.append(episode_reward)
            
            # Log every 10 episodes
            if self.episode_count % 10 == 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                if self.verbose > 0:
                    print(f"Episode {self.episode_count}: Mean Reward={mean_reward:.2f}")
        
        return True
    
    def get_metrics(self) -> Dict[str, float]:
        """Return collected metrics"""
        return {
            "total_episodes": float(self.episode_count),
            "mean_reward": float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
            "max_reward": float(np.max(self.episode_rewards)) if self.episode_rewards else 0.0,
            "min_reward": float(np.min(self.episode_rewards)) if self.episode_rewards else 0.0,
        }


def create_training_environment(n_laps: int = 50, num_envs: int = 1) -> Tuple[DummyVecEnv, pd.DataFrame]:
    """
    Create training environment(s) with telemetry data
    
    Args:
        n_laps: Number of laps in training data
        num_envs: Number of parallel environments
        
    Returns:
        Tuple of (vectorized environment, telemetry dataframe)
    """
    print(f"📊 Creating training environment ({num_envs} env(s))...")
    
    # Generate training telemetry
    gen = SyntheticTelemetry()
    session = gen.generate_race_session(n_laps=n_laps)
    telemetry_df = session.telemetry_df
    
    # Create environment(s)
    def make_env(rank: int = 0):
        def _init():
            env = MotoBioEnv(telemetry_df=telemetry_df)
            env = Monitor(env)
            return env
        return _init
    
    # Vectorize environments
    env = DummyVecEnv([make_env(i) for i in range(num_envs)])
    
    print(f"✅ Environment created with {len(telemetry_df)} observations")
    
    return env, telemetry_df


def train_ppo_agent(env: DummyVecEnv, total_timesteps: int = None,
                   save_dir: Path = None) -> Tuple[PPO, Dict[str, Any]]:
    """
    Train PPO agent on the bio-adaptive environment
    
    Args:
        env: Vectorized gym environment
        total_timesteps: Total training timesteps
        save_dir: Directory to save models and logs
        
    Returns:
        Tuple of (trained model, training metrics)
    """
    if total_timesteps is None:
        total_timesteps = TRAIN_CONFIG.TOTAL_TIMESTEPS
    if save_dir is None:
        save_dir = PATHS.MODELS_DIR
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 Starting PPO training...")
    print(f"   Total timesteps: {total_timesteps}")
    print(f"   Learning rate: {TRAIN_CONFIG.LEARNING_RATE}")
    print(f"   Network: {TRAIN_CONFIG.POLICY_NETWORK_LAYERS}")
    
    # Create PPO agent with simplified config
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=TRAIN_CONFIG.LEARNING_RATE,
        n_steps=TRAIN_CONFIG.N_STEPS,
        batch_size=TRAIN_CONFIG.BATCH_SIZE,
        n_epochs=TRAIN_CONFIG.N_EPOCHS,
        gamma=TRAIN_CONFIG.GAMMA,
        gae_lambda=TRAIN_CONFIG.GAE_LAMBDA,
        clip_range=TRAIN_CONFIG.CLIP_RANGE,
        ent_coef=TRAIN_CONFIG.ENTROPY_COEF,
        verbose=1,
        tensorboard_log=str(PATHS.LOGS_DIR),
    )
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, total_timesteps // 10),
        save_path=str(save_dir),
        name_prefix="ppo_checkpoint",
    )
    
    # Bio-adaptive callback
    bio_callback = BioAdaptiveCallback(verbose=1)
    
    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, bio_callback],
        log_interval=TRAIN_CONFIG.LOG_INTERVAL,
    )
    
    # Save final model
    model_path = save_dir / "ppo_bio_adaptive"
    model.save(str(model_path))
    print(f"✅ Model saved to {model_path}.zip")
    
    # Collect metrics
    metrics = bio_callback.get_metrics()
    metrics["total_timesteps"] = float(total_timesteps)
    metrics["learning_rate"] = float(TRAIN_CONFIG.LEARNING_RATE)
    
    # Save metrics
    metrics_path = PATHS.LOGS_DIR / "training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"📊 Training complete!")
    print(f"   Mean reward: {metrics['mean_reward']:.2f}")
    print(f"   Max reward: {metrics['max_reward']:.2f}")
    
    return model, metrics


def load_and_evaluate(model_path: str, env: MotoBioEnv,
                     n_episodes: int = 5) -> Dict[str, float]:
    """
    Load trained model and evaluate on environment
    
    Args:
        model_path: Path to saved model (.zip)
        env: Environment for evaluation
        n_episodes: Number of evaluation episodes
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n📈 Loading model and evaluating...")
    
    # Load model
    model = PPO.load(model_path)
    
    episode_rewards = []
    episode_lengths = []
    episode_bio_gates = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        bio_gates = 0
        
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            if info.get("was_bio_gated", False):
                bio_gates += 1
            
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        episode_bio_gates.append(bio_gates)
        
        print(f"   Episode {ep+1}: Reward={episode_reward:.2f}, Steps={episode_steps}, Bio-Gates={bio_gates}")
    
    metrics = {
        "mean_episode_reward": float(np.mean(episode_rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_bio_gate_activations": float(np.mean(episode_bio_gates)),
        "total_evaluation_episodes": float(n_episodes),
    }
    
    return metrics


def main():
    """Main training pipeline"""
    # Create environment
    env, telemetry_df = create_training_environment(n_laps=100, num_envs=1)
    
    # Train agent
    model, train_metrics = train_ppo_agent(
        env=env,
        total_timesteps=50000,
        save_dir=PATHS.MODELS_DIR
    )
    
    # Evaluate
    eval_env = MotoBioEnv(telemetry_df=telemetry_df)
    eval_metrics = load_and_evaluate(
        model_path=str(PATHS.MODELS_DIR / "ppo_bio_adaptive"),
        env=eval_env,
        n_episodes=5
    )
    
    print(f"\n📊 Final Metrics:")
    print(json.dumps({**train_metrics, **eval_metrics}, indent=2))


if __name__ == "__main__":
    main()
