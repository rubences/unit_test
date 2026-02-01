"""
Phase 3: PPO Training Loop with Bio-Adaptive Feedback

Train a Proximal Policy Optimization (PPO) agent on the MotoBioEnv.
This implements the reinforcement learning component of the bio-cybernetic system.

Key Features:
- Logs average reward and bio-gate activations per episode
- Implements a custom callback to track "Doctor vs Engineer" dynamics
- Saves trained model to disk
- Provides convergence diagnostics
"""

import numpy as np
import os
from pathlib import Path
from typing import Dict, List
import json

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from env import MotoBioEnv


class BioAdaptiveCallback(BaseCallback):
    """
    Custom callback to track bio-adaptive coaching metrics.
    
    Logs:
    - Average episode reward
    - Bio-gate activations per episode
    - Off-track events per episode
    - Policy entropy (exploration level)
    
    These metrics show the "Doctor vs Engineer" dynamic:
    - "Engineer" = Learned policy (trying to maximize speed)
    - "Doctor" = Bio-gating mechanism (enforcing safety)
    - Interaction = How often does the doctor override the engineer?
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_gate_activations = []
        self.episode_off_track_events = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_gates = 0
        self.current_episode_off_track = 0
        self.current_episode_length = 0
    
    def _on_step(self) -> bool:
        """Called after every environment step."""
        
        # Track metrics from the environment
        if self.model.env.envs[0].env.bio_gate_activations > self.current_episode_gates:
            self.current_episode_gates = self.model.env.envs[0].env.bio_gate_activations
        
        if self.model.env.envs[0].env.off_track_events > self.current_episode_off_track:
            self.current_episode_off_track = self.model.env.envs[0].env.off_track_events
        
        self.current_episode_reward += self.locals.get('rewards', [0])[0]
        self.current_episode_length += 1
        
        # Check if episode ended (done signal)
        if self.locals.get('dones', [False])[0]:
            # Episode ended, log metrics
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_gate_activations.append(self.current_episode_gates)
            self.episode_off_track_events.append(self.current_episode_off_track)
            self.episode_lengths.append(self.current_episode_length)
            
            # Reset accumulators
            self.current_episode_reward = 0.0
            self.current_episode_gates = 0
            self.current_episode_off_track = 0
            self.current_episode_length = 0
        
        return True
    
    def get_summary(self) -> Dict:
        """Return summary of training metrics."""
        if len(self.episode_rewards) == 0:
            return {}
        
        return {
            'avg_episode_reward': float(np.mean(self.episode_rewards)),
            'std_episode_reward': float(np.std(self.episode_rewards)),
            'avg_bio_gates_per_episode': float(np.mean(self.episode_gate_activations)),
            'avg_off_track_events_per_episode': float(np.mean(self.episode_off_track_events)),
            'avg_episode_length': float(np.mean(self.episode_lengths)),
            'total_episodes': len(self.episode_rewards),
            'total_timesteps': self.num_timesteps
        }


def train_ppo_agent(
    env: MotoBioEnv,
    total_timesteps: int = 10000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    output_dir: str = 'models'
) -> tuple:
    """
    Train a PPO agent on the bio-adaptive environment.
    
    PPO Hyperparameters (OpenAI defaults adapted for motorcycle racing):
    - learning_rate: 3e-4 (classic PPO learning rate)
    - n_steps: 2048 (trajectory length before update)
    - batch_size: 64 (mini-batch size)
    - n_epochs: 10 (policy update epochs)
    - gamma: 0.99 (discount factor, high because racing is long-horizon)
    - gae_lambda: 0.95 (GAE advantage estimation parameter)
    
    Args:
        env: MotoBioEnv instance
        total_timesteps: Total training timesteps
        learning_rate: Actor-critic learning rate
        n_steps: Rollout length (trajectory before policy update)
        batch_size: Mini-batch size for policy updates
        n_epochs: Number of policy update epochs per rollout
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        output_dir: Where to save trained model
    
    Returns:
        tuple: (trained_model, callback_metrics)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("PPO TRAINING: Bio-Adaptive Motorcycle Racing")
    print("=" * 70)
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Batch Size: {batch_size}")
    print(f"Gamma (Discount): {gamma}")
    print("=" * 70)
    
    # Create callback
    callback = BioAdaptiveCallback(verbose=1)
    
    # Initialize PPO model
    model = PPO(
        'MlpPolicy',  # Multi-layer perceptron policy
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        verbose=1,
        device='cpu',  # Use CPU for reproducibility
        tensorboard_log='./logs'
    )
    
    # Configure logger
    logger = configure('./logs', ["csv", "tensorboard"])
    model.set_logger(logger)
    
    # Train
    print(f"\nStarting training ({total_timesteps:,} timesteps)...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    # Save trained model
    model_path = os.path.join(output_dir, 'ppo_bio_adaptive')
    model.save(model_path)
    print(f"\n✓ Model saved to {model_path}.zip")
    
    # Get training metrics
    metrics = callback.get_summary()
    
    # Save metrics to JSON
    metrics_path = os.path.join(output_dir, 'training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Training metrics saved to {metrics_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Total Episodes: {metrics['total_episodes']}")
    print(f"Avg Episode Reward: {metrics['avg_episode_reward']:.4f} ± {metrics['std_episode_reward']:.4f}")
    print(f"Avg Bio-Gate Activations/Episode: {metrics['avg_bio_gates_per_episode']:.2f}")
    print(f"Avg Off-Track Events/Episode: {metrics['avg_off_track_events_per_episode']:.2f}")
    print(f"Avg Episode Length: {metrics['avg_episode_length']:.1f} steps")
    print("=" * 70)
    
    # Key insight: Doctor vs Engineer
    print("\n📊 DOCTOR vs ENGINEER DYNAMICS:")
    print(f"  Engineer (learned policy) suggested actions: {metrics['total_timesteps']} times")
    total_gates = metrics['avg_bio_gates_per_episode'] * metrics['total_episodes']
    print(f"  Doctor (bio-gate) overrode with safety: {total_gates:.0f} times")
    override_rate = 100 * (total_gates / max(1, metrics['total_timesteps']))
    print(f"  Override Rate: {override_rate:.2f}%")
    print(f"\n  Interpretation:")
    print(f"  - High override rate: Policy is learning aggressive behavior")
    print(f"  - Low override rate: Policy respects stress limits")
    print()
    
    return model, metrics


def load_and_evaluate(
    model_path: str,
    env: MotoBioEnv,
    n_episodes: int = 3,
    render: bool = False
) -> Dict:
    """
    Load a trained model and evaluate on the environment.
    
    Args:
        model_path: Path to saved model (without .zip)
        env: MotoBioEnv instance
        n_episodes: Number of evaluation episodes
        render: Whether to print observations
    
    Returns:
        dict: Evaluation metrics
    """
    
    print(f"\nLoading model from {model_path}...")
    model = PPO.load(model_path)
    print("✓ Model loaded")
    
    print(f"Evaluating for {n_episodes} episodes...")
    
    episode_rewards = []
    episode_gates = []
    episode_off_track = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        gates_this_ep = info['bio_gate_activations']
        off_track_this_ep = info['off_track_events']
        
        done = False
        step = 0
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            step += 1
        
        gates_activated = info.get('bio_gate_activations', 0) - gates_this_ep
        off_track_events = info.get('off_track_events', 0) - off_track_this_ep
        
        episode_rewards.append(episode_reward)
        episode_gates.append(gates_activated)
        episode_off_track.append(off_track_events)
        
        print(f"  Episode {ep+1}: Reward={episode_reward:.4f}, Gates={gates_activated}, Off-Track={off_track_events}")
    
    eval_metrics = {
        'avg_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'avg_gates_per_episode': float(np.mean(episode_gates)),
        'avg_off_track_per_episode': float(np.mean(episode_off_track)),
        'episodes_evaluated': n_episodes
    }
    
    return eval_metrics


if __name__ == '__main__':
    # Create environment
    print("Creating environment...")
    env = MotoBioEnv(episode_length=600)  # 60 seconds at 10 Hz
    print(f"✓ Environment created")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # Train
    model, metrics = train_ppo_agent(
        env,
        total_timesteps=10000,
        output_dir='models'
    )
    
    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)
    eval_metrics = load_and_evaluate('models/ppo_bio_adaptive', env, n_episodes=3)
    
    print("\nEvaluation Summary:")
    print(f"  Avg Reward: {eval_metrics['avg_reward']:.4f}")
    print(f"  Avg Bio-Gates/Episode: {eval_metrics['avg_gates_per_episode']:.2f}")
    print(f"  Avg Off-Track/Episode: {eval_metrics['avg_off_track_per_episode']:.2f}")
    
    env.close()
    print("\n✓ Training and evaluation complete!")
