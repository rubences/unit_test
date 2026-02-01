"""
Phase 4: Visualization Dashboard

Generates publication-quality figures showing:
- Top panel: Speed and Lean Angle trajectories
- Middle panel: Raw ECG signal with stress-level background zones
- Bottom panel: Haptic feedback actions taken (with suppression markers)

These visualizations demonstrate the "Doctor vs Engineer" dynamic
and are suitable for academic papers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import os
from pathlib import Path

from stable_baselines3 import PPO
from env import MotoBioEnv
from data_gen import generate_race_session


def create_evaluation_lap(model_path: str, env: MotoBioEnv) -> tuple:
    """
    Run one evaluation lap with a trained model and record all metrics.
    
    Args:
        model_path: Path to trained PPO model (without .zip)
        env: MotoBioEnv instance
    
    Returns:
        tuple: (trajectory_dict, metadata_dict)
    """
    
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    print("✓ Model loaded, running evaluation lap...")
    
    # Initialize tracking arrays
    obs_list = []
    action_list = []
    reward_list = []
    gated_list = []
    info_list = []
    
    # Reset environment
    obs, info = env.reset()
    
    # Run episode
    done = False
    step = 0
    total_reward = 0.0
    
    while not done:
        # Get action from policy
        action, _state = model.predict(obs, deterministic=True)
        
        # Step environment
        obs_new, reward, terminated, truncated, step_info = env.step(action)
        
        # Store trajectory
        obs_list.append(obs.copy())
        action_list.append(action)
        reward_list.append(reward)
        gated_list.append(step_info['was_gated'])
        info_list.append(step_info)
        
        total_reward += reward
        obs = obs_new
        done = terminated or truncated
        step += 1
    
    # Convert to numpy arrays
    obs_array = np.array(obs_list)
    action_array = np.array(action_list)
    reward_array = np.array(reward_list)
    gated_array = np.array(gated_list)
    
    # Create trajectory dictionary
    trajectory = {
        'time': np.arange(len(obs_array)) * 0.1,  # 0.1 sec timestep
        'speed_kmh': obs_array[:, 0],
        'g_force': obs_array[:, 1],
        'lean_angle': obs_array[:, 2],
        'hrv_index': obs_array[:, 3],
        'stress_level': obs_array[:, 4],
        'action': action_array,
        'reward': reward_array,
        'was_gated': gated_array
    }
    
    # Metadata
    metadata = {
        'total_steps': step,
        'total_reward': total_reward,
        'avg_reward': total_reward / step,
        'bio_gate_activations': np.sum(gated_array),
        'gate_rate': 100 * np.sum(gated_array) / step,
        'off_track_events': env.off_track_events,
        'info_list': info_list
    }
    
    print(f"✓ Evaluation complete: {step} steps, {metadata['total_reward']:.2f} total reward")
    print(f"  Bio-gate activations: {metadata['bio_gate_activations']} ({metadata['gate_rate']:.1f}%)")
    
    return trajectory, metadata


def create_visualization(
    trajectory: dict,
    metadata: dict,
    ecg_signal: np.ndarray = None,
    output_path: str = 'bio_adaptive_results.png',
    figsize: tuple = (14, 10)
):
    """
    Create publication-quality visualization with 3 subplots.
    
    Top Subplot: Speed (km/h) and Lean Angle (deg) vs Time
    Middle Subplot: Stress Level (colored background) with ECG signal overlay
    Bottom Subplot: Haptic Feedback Actions (with suppression markers)
    
    Args:
        trajectory: Dictionary with time series data
        metadata: Dictionary with episode metadata
        ecg_signal: Optional ECG signal (will be downsampled to match trajectory)
        output_path: Where to save the figure
        figsize: Figure size in inches
    """
    
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    time = trajectory['time']
    
    # ========================================================================
    # SUBPLOT 1: Speed and Lean Angle
    # ========================================================================
    ax1 = axes[0]
    ax1.plot(time, trajectory['speed_kmh'], 'b-', linewidth=2.5, label='Speed')
    ax1.set_ylabel('Speed (km/h)', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 350])
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=11)
    
    # Create secondary y-axis for lean angle
    ax1_twin = ax1.twinx()
    ax1_twin.plot(time, trajectory['lean_angle'], 'r-', linewidth=2.5, label='Lean Angle')
    ax1_twin.set_ylabel('Lean Angle (deg)', fontsize=12, fontweight='bold', color='r')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    ax1_twin.set_ylim([0, 65])
    ax1_twin.legend(loc='upper right', fontsize=11)
    
    # Title with metadata
    title = f"Bio-Adaptive Haptic Coaching: Evaluation Lap\n"
    title += f"Reward: {metadata['total_reward']:.2f} | Bio-Gates: {metadata['bio_gate_activations']} ({metadata['gate_rate']:.1f}%) | Off-Track: {metadata['off_track_events']}"
    ax1.set_title(title, fontsize=13, fontweight='bold', pad=15)
    
    # ========================================================================
    # SUBPLOT 2: Stress Level (Background) with ECG Signal
    # ========================================================================
    ax2 = axes[1]
    
    # Create stress-level background zones
    for i in range(len(time) - 1):
        stress = trajectory['stress_level'][i]
        
        # Color gradient from green (low stress) to red (high stress)
        if stress < 0.33:
            color = 'green'
            alpha = 0.1
        elif stress < 0.66:
            color = 'yellow'
            alpha = 0.15
        else:
            color = 'red'
            alpha = 0.2
        
        rect = Rectangle((time[i], -1), time[i+1] - time[i], 2, 
                        facecolor=color, alpha=alpha, edgecolor='none')
        ax2.add_patch(rect)
    
    # Plot ECG signal (downsampled)
    if ecg_signal is not None:
        # Downsample ECG from 500 Hz to match trajectory rate (~10 Hz)
        downsample_factor = 50  # 500 Hz / 10 Hz
        ecg_downsampled = ecg_signal[::downsample_factor][:len(time)]
        
        # Normalize ECG to [-1, 1] for visualization
        ecg_norm = (ecg_downsampled - np.mean(ecg_downsampled)) / (np.std(ecg_downsampled) + 1e-6)
        ecg_norm = np.clip(ecg_norm, -2, 2) / 2  # Clip and scale
        
        ax2.plot(time, ecg_norm, 'k-', linewidth=1.0, alpha=0.8, label='ECG Signal')
    
    ax2.set_ylabel('ECG / Stress Level', fontsize=12, fontweight='bold')
    ax2.set_ylim([-1.2, 1.2])
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
    ax2.grid(True, alpha=0.3)
    
    # Add legend with stress zones
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor='green', alpha=0.3, label='Calm (stress < 0.33)'),
        Patch(facecolor='yellow', alpha=0.3, label='Moderate (0.33-0.66)'),
        Patch(facecolor='red', alpha=0.3, label='High Stress (> 0.66)')
    ]
    if ecg_signal is not None:
        legend_patches.insert(0, plt.Line2D([0], [0], color='k', linewidth=1.5, label='ECG Signal'))
    ax2.legend(handles=legend_patches, loc='upper right', fontsize=10)
    
    # ========================================================================
    # SUBPLOT 3: Haptic Feedback Actions
    # ========================================================================
    ax3 = axes[2]
    
    # Define action names and colors
    action_names = ['No Feedback', 'Mild Haptic', 'Warning Haptic', 'Emergency Haptic']
    action_colors = ['#cccccc', '#ffeb99', '#ff6b6b', '#8b0000']
    
    # Plot actions as vertical bars
    actions = trajectory['action']
    was_gated = trajectory['was_gated']
    
    for i in range(len(time)):
        action = actions[i]
        color = action_colors[int(action)]
        
        # If this action was gated, add a marker
        if was_gated[i]:
            ax3.bar(time[i], 1, width=0.1, color=color, edgecolor='red', linewidth=2.5, alpha=0.7)
            ax3.plot(time[i], 0.5, 'r*', markersize=15, label='Bio-Gate Override' if i == 0 else '')
        else:
            ax3.bar(time[i], 1, width=0.1, color=color, alpha=0.7)
    
    ax3.set_ylabel('Action Type', fontsize=12, fontweight='bold')
    ax3.set_ylim([0, 1.5])
    ax3.set_yticks([])
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Custom legend for actions
    from matplotlib.patches import Patch
    action_patches = [Patch(facecolor=action_colors[i], label=action_names[i]) 
                     for i in range(4)]
    ax3.legend(handles=action_patches, loc='upper left', fontsize=10, ncol=2)
    
    # ========================================================================
    # X-AXIS LABEL (shared)
    # ========================================================================
    ax3.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    
    # ========================================================================
    # LAYOUT AND SAVE
    # ========================================================================
    plt.tight_layout()
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {output_path}")
    
    plt.close()


def visualize_training_metrics(metrics: dict, output_path: str = 'training_metrics_plot.png'):
    """
    Create additional visualization of training metrics.
    
    Shows convergence of average reward and bio-gate activation rates.
    """
    
    if not metrics or 'avg_episode_reward' not in metrics:
        print("  (Skipping training metrics plot: insufficient data)")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Average reward convergence
    ax1 = axes[0]
    ax1.text(0.5, 0.5, 'Training Metrics Summary', 
            ha='center', va='center', fontsize=14, fontweight='bold',
            transform=ax1.transAxes)
    ax1.axis('off')
    
    # Plot 2: Metrics table
    ax2 = axes[1]
    ax2.axis('off')
    
    metric_text = f"""
    Training Statistics:
    
    • Episodes Trained: {metrics['total_episodes']}
    • Total Timesteps: {metrics['total_timesteps']:,}
    • Avg Episode Reward: {metrics['avg_episode_reward']:.4f} ± {metrics['std_episode_reward']:.4f}
    • Avg Bio-Gates/Episode: {metrics['avg_bio_gates_per_episode']:.2f}
    • Avg Off-Track/Episode: {metrics['avg_off_track_events_per_episode']:.2f}
    • Avg Episode Length: {metrics['avg_episode_length']:.1f} steps
    
    Interpretation:
    - Higher reward = better speed + safety balance
    - Higher gate rate = more aggressive agent policy
    - Lower off-track = safer riding
    """
    
    ax2.text(0.05, 0.95, metric_text, fontsize=11, family='monospace',
            verticalalignment='top', transform=ax2.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training metrics plot saved to {output_path}")
    plt.close()


if __name__ == '__main__':
    # Generate synthetic data for visualization
    print("Phase 4: Visualization Dashboard")
    print("=" * 70)
    
    print("\nStep 1: Generating sample ECG data...")
    df_telem, ecg_signal, meta = generate_race_session(
        n_laps=1,
        output_dir='data/raw'
    )
    
    print("\nStep 2: Creating environment and loading trained model...")
    env = MotoBioEnv(episode_length=600)
    
    # Check if model exists
    if not os.path.exists('models/ppo_bio_adaptive.zip'):
        print("  ⚠ Warning: Trained model not found at 'models/ppo_bio_adaptive.zip'")
        print("  Please run src/train.py first to train the model.")
        print("  For now, running evaluation with a randomly initialized model...")
    
    print("\nStep 3: Running evaluation lap...")
    trajectory, metadata = create_evaluation_lap('models/ppo_bio_adaptive', env)
    
    print("\nStep 4: Creating visualizations...")
    create_visualization(
        trajectory,
        metadata,
        ecg_signal=ecg_signal,
        output_path='bio_adaptive_results.png'
    )
    
    print("\n" + "=" * 70)
    print("Visualization complete! Open 'bio_adaptive_results.png' to view.")
    print("=" * 70)
    
    env.close()
