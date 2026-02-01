"""
Visualization Module for Bio-Adaptive Results
Generates publication-quality 3-panel dashboard with ECG, stress zones, and haptic actions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Dict, Any, Tuple
import json

# Soporte para importación relativa y absoluta
try:
    from .config import VIS_CONFIG, PATHS, SIM_CONFIG
    from .environment import MotoBioEnv
    from .data_gen import SyntheticTelemetry
except ImportError:
    from config import VIS_CONFIG, PATHS, SIM_CONFIG
    from environment import MotoBioEnv
    from data_gen import SyntheticTelemetry

from stable_baselines3 import PPO


def create_evaluation_lap(model_path: str, env: MotoBioEnv,
                         telemetry_df: pd.DataFrame = None) -> Tuple[Dict, np.ndarray]:
    """
    Run one evaluation lap with trained model and record trajectory
    
    Args:
        model_path: Path to trained PPO model
        env: MotoBioEnv instance
        telemetry_df: Optional telemetry data
        
    Returns:
        Tuple of (trajectory dict, ECG signal)
    """
    print("🏁 Running evaluation lap...")
    
    # Load model
    model = PPO.load(model_path)
    
    # Initialize storage
    trajectory = {
        'time': [],
        'speed': [],
        'lean_angle': [],
        'g_force': [],
        'heart_rate': [],
        'stress': [],
        'action': [],
        'reward': [],
        'bio_gated': [],
    }
    
    # Reset environment
    obs, info = env.reset()
    done = False
    step = 0
    
    # Load ECG data if available
    ecg_file = PATHS.DATA_DIR / "ecg_signal.npy"
    if ecg_file.exists():
        ecg_signal = np.load(ecg_file)
    else:
        ecg_signal = None
    
    # Run evaluation
    while not done and step < 5000:
        # Get prediction
        action, _ = model.predict(obs, deterministic=True)
        
        # Store current state
        trajectory['time'].append(step / SIM_CONFIG.SAMPLING_RATE_TELEMETRY)
        trajectory['speed'].append(info['speed_kmh'])
        trajectory['heart_rate'].append(info['heart_rate_bpm'])
        trajectory['action'].append(action)
        trajectory['bio_gated'].append(info.get('was_bio_gated', False))
        
        # Extract from observation
        trajectory['speed'].append(obs[0] * SIM_CONFIG.MAX_SPEED_KMH)
        trajectory['lean_angle'].append(obs[1] * SIM_CONFIG.MAX_LEAN_ANGLE)
        trajectory['g_force'].append(obs[2] * SIM_CONFIG.MAX_G_FORCE)
        trajectory['stress'].append(obs[4])
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        trajectory['reward'].append(reward)
        
        done = terminated or truncated
        step += 1
    
    # Convert to arrays
    for key in trajectory:
        trajectory[key] = np.array(trajectory[key])
    
    print(f"✅ Evaluation complete: {step} steps")
    
    return trajectory, ecg_signal


def create_visualization(trajectory: Dict, telemetry_df: pd.DataFrame = None,
                        ecg_signal: np.ndarray = None,
                        output_path: str = None) -> None:
    """
    Create 3-panel publication-quality dashboard
    
    Panel 1 (Top): Speed and Lean Angle trajectories
    Panel 2 (Middle): ECG signal with stress-level background color zones
    Panel 3 (Bottom): Haptic actions with bio-gate markers
    
    Args:
        trajectory: Dictionary with trajectory data
        telemetry_df: Optional telemetry for reference
        ecg_signal: ECG signal array
        output_path: Where to save figure
    """
    if output_path is None:
        output_path = PATHS.LOGS_DIR / "bio_adaptive_results.png"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Creating visualization...")
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=VIS_CONFIG.FIGURE_SIZE, dpi=VIS_CONFIG.FIGURE_DPI)
    fig.suptitle(
        'Bio-Adaptive Haptic Coaching System: Evaluation Results',
        fontsize=VIS_CONFIG.FONT_SIZE_TITLE,
        fontweight='bold'
    )
    
    time = trajectory['time']
    speed = trajectory['speed']
    lean_angle = trajectory['lean_angle']
    stress = trajectory['stress']
    action = trajectory['action']
    bio_gated = trajectory['bio_gated']
    
    # ==================== PANEL 1: Speed & Lean Angle ====================
    ax1 = axes[0]
    
    ax1_twin = ax1.twinx()
    line1 = ax1.plot(time, speed, 'b-', linewidth=2.5, label='Speed')
    line2 = ax1_twin.plot(time, lean_angle, 'r-', linewidth=2.5, label='Lean Angle')
    
    ax1.set_xlabel('Time (s)', fontsize=VIS_CONFIG.FONT_SIZE_LABEL)
    ax1.set_ylabel('Speed (km/h)', fontsize=VIS_CONFIG.FONT_SIZE_LABEL, color='b')
    ax1_twin.set_ylabel('Lean Angle (°)', fontsize=VIS_CONFIG.FONT_SIZE_LABEL, color='r')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    ax1.grid(True, alpha=0.3)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=VIS_CONFIG.FONT_SIZE_LEGEND)
    ax1.set_title('Speed & Lean Angle Trajectory', fontsize=VIS_CONFIG.FONT_SIZE_LABEL, pad=10)
    
    # ==================== PANEL 2: ECG with Stress Zones ====================
    ax2 = axes[1]
    
    # Background stress zones
    for i in range(len(time) - 1):
        stress_val = stress[i]
        if stress_val < SIM_CONFIG.MODERATE_STRESS_THRESHOLD:
            color = VIS_CONFIG.COLOR_GREEN_CALM
        elif stress_val < SIM_CONFIG.HIGH_STRESS_THRESHOLD:
            color = VIS_CONFIG.COLOR_YELLOW_MODERATE
        else:
            color = VIS_CONFIG.COLOR_RED_PANIC
        
        ax2.axvspan(time[i], time[i+1], alpha=0.2, color=color)
    
    # ECG signal
    if ecg_signal is not None:
        ecg_time = np.linspace(0, max(time), len(ecg_signal))
        # Normalize ECG for visualization
        ecg_normalized = (ecg_signal - np.min(ecg_signal)) / (np.max(ecg_signal) - np.min(ecg_signal))
        ecg_normalized = ecg_normalized * 100  # Scale for visibility
        
        ax2.plot(ecg_time, ecg_normalized, 'k-', linewidth=1.0, alpha=0.7, label='ECG Signal')
        ax2.set_ylabel('ECG (normalized)', fontsize=VIS_CONFIG.FONT_SIZE_LABEL)
    else:
        ax2.text(0.5, 0.5, 'No ECG data available', ha='center', va='center',
                transform=ax2.transAxes, fontsize=12)
    
    ax2.set_xlabel('Time (s)', fontsize=VIS_CONFIG.FONT_SIZE_LABEL)
    ax2.set_title('ECG Signal with Stress Zones (Green=Calm, Yellow=Moderate, Red=Panic)',
                 fontsize=VIS_CONFIG.FONT_SIZE_LABEL, pad=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=VIS_CONFIG.FONT_SIZE_LEGEND)
    
    # ==================== PANEL 3: Haptic Actions & Bio-Gate ====================
    ax3 = axes[2]
    
    # Action levels with colors
    action_colors = {
        0: '#2ecc71',  # No Feedback - Green
        1: '#3498db',  # Mild - Blue
        2: '#f39c12',  # Warning - Orange
        3: '#e74c3c',  # Emergency - Red
    }
    action_labels = {
        0: 'No Feedback',
        1: 'Mild Haptic',
        2: 'Warning Haptic',
        3: 'Emergency Haptic',
    }
    
    for i in range(len(time) - 1):
        act = int(action[i])
        color = action_colors.get(act, 'gray')
        
        # Draw bar for action
        ax3.barh(0, time[i+1] - time[i], left=time[i], height=0.6,
                color=color, edgecolor='black', linewidth=0.5, alpha=0.8)
        
        # Mark bio-gate activations with red border
        if bio_gated[i]:
            rect = patches.Rectangle(
                (time[i], -0.3), time[i+1] - time[i], 0.6,
                linewidth=2, edgecolor='red', facecolor='none',
                linestyle='--', alpha=VIS_CONFIG.BIOGATE_ALPHA
            )
            ax3.add_patch(rect)
    
    ax3.set_xlim(0, max(time))
    ax3.set_ylim(-0.5, 0.5)
    ax3.set_xlabel('Time (s)', fontsize=VIS_CONFIG.FONT_SIZE_LABEL)
    ax3.set_yticks([])
    ax3.set_title('Haptic Actions (Red Border = Bio-Gate Override)', 
                 fontsize=VIS_CONFIG.FONT_SIZE_LABEL, pad=10)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Custom legend for actions
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=action_colors[i], edgecolor='black', label=action_labels[i])
        for i in range(4)
    ]
    ax3.legend(handles=legend_elements, loc='upper center', ncol=4,
              fontsize=VIS_CONFIG.FONT_SIZE_LEGEND, bbox_to_anchor=(0.5, -0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=VIS_CONFIG.FIGURE_DPI, bbox_inches='tight')
    print(f"✅ Visualization saved to {output_path}")
    
    plt.close()


def visualize_training_metrics(metrics: Dict, output_path: str = None) -> None:
    """
    Create training metrics summary plot
    
    Args:
        metrics: Training metrics dictionary
        output_path: Where to save figure
    """
    if output_path is None:
        output_path = PATHS.LOGS_DIR / "training_metrics_plot.png"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📈 Creating training metrics plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=VIS_CONFIG.FIGURE_DPI)
    
    # Create bar chart
    keys = list(metrics.keys())
    values = list(metrics.values())
    
    # Filter numeric values
    numeric_keys = []
    numeric_values = []
    for k, v in zip(keys, values):
        if isinstance(v, (int, float)):
            numeric_keys.append(k)
            numeric_values.append(v)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(numeric_keys)))
    bars = ax.bar(range(len(numeric_keys)), numeric_values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Labels
    ax.set_xticks(range(len(numeric_keys)))
    ax.set_xticklabels(numeric_keys, rotation=45, ha='right')
    ax.set_ylabel('Value', fontsize=VIS_CONFIG.FONT_SIZE_LABEL)
    ax.set_title('Training Metrics Summary', fontsize=VIS_CONFIG.FONT_SIZE_TITLE, fontweight='bold')
    
    # Add value labels on bars
    for bar, val in zip(bars, numeric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=VIS_CONFIG.FIGURE_DPI, bbox_inches='tight')
    print(f"✅ Metrics plot saved to {output_path}")
    
    plt.close()


def save_evaluation_report(trajectory: Dict, metrics: Dict, output_path: str = None) -> None:
    """
    Save detailed evaluation report as JSON
    
    Args:
        trajectory: Evaluation trajectory
        metrics: Evaluation metrics
        output_path: Where to save report
    """
    if output_path is None:
        output_path = PATHS.LOGS_DIR / "evaluation_metrics.json"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        "trajectory_summary": {
            "mean_speed_kmh": float(np.mean(trajectory['speed'])),
            "max_speed_kmh": float(np.max(trajectory['speed'])),
            "mean_lean_angle_deg": float(np.mean(trajectory['lean_angle'])),
            "max_lean_angle_deg": float(np.max(trajectory['lean_angle'])),
            "mean_stress": float(np.mean(trajectory['stress'])),
            "max_stress": float(np.max(trajectory['stress'])),
            "bio_gate_activations": int(np.sum(trajectory['bio_gated'])),
            "total_steps": int(len(trajectory['time'])),
        },
        "metrics": metrics,
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Evaluation report saved to {output_path}")


def main():
    """Generate visualization from saved model"""
    # Generate data
    gen = SyntheticTelemetry()
    session = gen.generate_race_session(n_laps=20)
    
    # Create environment
    env = MotoBioEnv(telemetry_df=session.telemetry_df)
    
    # Run evaluation
    model_path = str(PATHS.MODELS_DIR / "ppo_bio_adaptive")
    trajectory, ecg = create_evaluation_lap(model_path, env, session.telemetry_df)
    
    # Create visualizations
    create_visualization(trajectory, session.telemetry_df, ecg)
    
    # Load metrics
    metrics_file = PATHS.LOGS_DIR / "training_metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
        visualize_training_metrics(metrics)
    
    print("✅ Visualization complete!")


if __name__ == "__main__":
    main()
