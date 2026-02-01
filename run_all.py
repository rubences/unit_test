#!/usr/bin/env python3
"""
run_all.py: Complete Pipeline Orchestrator

Executes all 4 phases of the Bio-Adaptive Haptic Coaching PoC:

🧬 Phase 1: Data Generation
   └─ Synthetic telemetry (10 laps) + ECG signals
   
🏍️ Phase 2: Environment Setup
   └─ MotoBioEnv with bio-gating mechanism
   
🧠 Phase 3: Model Training
   └─ PPO training for 10,000 timesteps
   
📊 Phase 4: Visualization
   └─ Publication-quality figures

Total runtime: ~5-10 minutes (depending on system)
Output: Trained model + Visualization figures
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_gen import generate_race_session
from env import MotoBioEnv
from train import train_ppo_agent, load_and_evaluate
from vis import create_evaluation_lap, create_visualization, visualize_training_metrics

import json
from datetime import datetime


def print_header(phase: int, title: str):
    """Print formatted section header."""
    width = 80
    print("\n" + "=" * width)
    print(f"{'🔹' * 2} PHASE {phase}: {title}".ljust(width - 4) + "{'🔹' * 2}")
    print("=" * width)


def print_section(text: str):
    """Print formatted section."""
    print(f"\n📍 {text}")
    print("-" * 70)


def main():
    """Execute complete pipeline."""
    
    print("\n" + "🚀" * 40)
    print("BIO-ADAPTIVE HAPTIC COACHING: PROOF-OF-CONCEPT")
    print("Complete Pipeline Execution")
    print("🚀" * 40)
    
    start_time = datetime.now()
    
    # ========================================================================
    # PHASE 1: DATA GENERATION
    # ========================================================================
    print_header(1, "Synthetic Data Generation")
    
    print_section("Generating 10-lap race session with realistic telemetry & ECG")
    print("This simulates a single rider through a complete race session")
    print("with physics-based dynamics and physiological correlations.\n")
    
    try:
        df_telemetry, ecg_signal, metadata = generate_race_session(
            n_laps=10,
            sampling_rate_telemetry=100,
            lap_duration=60,
            output_dir='data/raw'
        )
        print("\n✅ Phase 1 Complete: Data generation successful")
        print(f"   • Telemetry shape: {df_telemetry.shape}")
        print(f"   • ECG signal shape: {ecg_signal.shape}")
        print(f"   • Average HR: {metadata['avg_hr_bpm']:.1f} bpm")
        print(f"   • Average Speed: {metadata['avg_speed_kmh']:.1f} km/h")
    except Exception as e:
        print(f"\n❌ Phase 1 Failed: {e}")
        print("Please ensure numpy, pandas, and neurokit2 are installed.")
        return False
    
    # ========================================================================
    # PHASE 2: ENVIRONMENT SETUP
    # ========================================================================
    print_header(2, "Bio-Physics Environment Setup")
    
    print_section("Creating MotoBioEnv with bio-gating mechanism")
    print("This environment implements the closed-loop feedback system:")
    print("  Rider Physiology (RMSSD, HR) ←→ Learned Policy ←→ Haptic Feedback")
    print("  with non-learnable safety gating at the firmware level.\n")
    
    try:
        env = MotoBioEnv(
            telemetry_df=df_telemetry,
            episode_length=600  # 60 seconds at 10 Hz
        )
        
        obs, info = env.reset()
        print(f"✅ Environment created successfully")
        print(f"   • Observation space: {env.observation_space.shape}")
        print(f"   • Action space: {env.action_space.n} discrete actions")
        print(f"   • Initial observation: {obs}")
        
        # Test a few steps
        print_section("Testing environment with random actions")
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"   Step {step+1}: action={info['action']}, reward={reward:.4f}, stress={info['stress_level']:.2f}")
        
        obs, info = env.reset()  # Reset for training
        print(f"\n✅ Phase 2 Complete: Environment verified")
    except Exception as e:
        print(f"\n❌ Phase 2 Failed: {e}")
        print("Please ensure gymnasium and stable_baselines3 are installed.")
        return False
    
    # ========================================================================
    # PHASE 3: MODEL TRAINING
    # ========================================================================
    print_header(3, "PPO Training Loop")
    
    print_section("Training PPO agent for 10,000 timesteps")
    print("This implements the 'Engineer' component (learned policy).")
    print("Training will show how often the 'Doctor' (bio-gate) overrides.\n")
    
    try:
        model, training_metrics = train_ppo_agent(
            env=env,
            total_timesteps=10000,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            output_dir='models'
        )
        
        print(f"\n✅ Phase 3 Complete: Training successful")
        print(f"   • Model saved to: models/ppo_bio_adaptive.zip")
        print(f"   • Training metrics saved to: models/training_metrics.json")
        
        # Print key metrics
        print(f"\n   Training Summary:")
        print(f"   • Episodes: {training_metrics['total_episodes']}")
        print(f"   • Avg Reward: {training_metrics['avg_episode_reward']:.4f}")
        print(f"   • Bio-Gate Activations: {training_metrics['avg_bio_gates_per_episode']:.2f}/ep")
        print(f"   • Off-Track Events: {training_metrics['avg_off_track_events_per_episode']:.2f}/ep")
        
    except Exception as e:
        print(f"\n❌ Phase 3 Failed: {e}")
        print(f"Traceback: {type(e).__name__}: {str(e)}")
        return False
    
    # ========================================================================
    # PHASE 4: VISUALIZATION
    # ========================================================================
    print_header(4, "Visualization Dashboard")
    
    print_section("Running evaluation lap and generating publication-quality figures")
    print("Figures will show:")
    print("  • Top: Speed and Lean Angle trajectories")
    print("  • Middle: ECG signal with stress-level background zones")
    print("  • Bottom: Haptic feedback actions with suppression markers\n")
    
    try:
        # Evaluation lap
        trajectory, eval_metadata = create_evaluation_lap(
            'models/ppo_bio_adaptive',
            env
        )
        
        # Create main visualization
        create_visualization(
            trajectory=trajectory,
            metadata=eval_metadata,
            ecg_signal=ecg_signal,
            output_path='bio_adaptive_results.png'
        )
        
        # Create training metrics plot
        visualize_training_metrics(
            metrics=training_metrics,
            output_path='training_metrics_plot.png'
        )
        
        print(f"\n✅ Phase 4 Complete: Visualization successful")
        print(f"   • Main figure: bio_adaptive_results.png (300 DPI)")
        print(f"   • Metrics figure: training_metrics_plot.png")
        
    except Exception as e:
        print(f"\n❌ Phase 4 Failed: {e}")
        print(f"Traceback: {type(e).__name__}: {str(e)}")
        print("Note: Visualization can fail if model training was incomplete.")
        return False
    
    # ========================================================================
    # COMPLETION SUMMARY
    # ========================================================================
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    print_header(0, "PIPELINE COMPLETE ✨")
    
    print("\n📊 FINAL SUMMARY:")
    print("-" * 70)
    
    print(f"\n✅ All 4 Phases Completed Successfully!")
    print(f"\n⏱️  Total Runtime: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    
    print(f"\n📁 Output Files:")
    print(f"   Data:")
    print(f"   • data/raw/race_telemetry.csv - Telemetry data (10 laps)")
    print(f"   • data/raw/race_ecg.npz - ECG signals (raw)")
    print(f"   • data/raw/metadata.txt - Session metadata")
    print(f"\n   Models:")
    print(f"   • models/ppo_bio_adaptive.zip - Trained PPO agent")
    print(f"   • models/training_metrics.json - Training statistics")
    print(f"\n   Visualizations:")
    print(f"   • bio_adaptive_results.png - Main evaluation dashboard (300 DPI)")
    print(f"   • training_metrics_plot.png - Training convergence")
    print(f"\n   Logs:")
    print(f"   • logs/ - TensorBoard logs (view with: tensorboard --logdir logs)")
    
    print(f"\n🎯 Key Results:")
    print(f"   • Average Reward: {training_metrics['avg_episode_reward']:.4f}")
    print(f"   • Bio-Gate Override Rate: {training_metrics['avg_bio_gates_per_episode']:.2f}/episode")
    print(f"   • Safety (Off-Track): {training_metrics['avg_off_track_events_per_episode']:.2f}/episode")
    
    print(f"\n📖 Next Steps:")
    print(f"   1. Open 'bio_adaptive_results.png' to review the visualization")
    print(f"   2. Analyze the ECG signal patterns and stress zones")
    print(f"   3. Study how the bio-gate mechanism suppresses feedback during high stress")
    print(f"   4. Use trained model for policy analysis and ablation studies")
    print(f"   5. Adapt code for real-world motorcycle telemetry data")
    
    print(f"\n✨ Paper Validation:")
    print(f"   This PoC demonstrates all components from the academic paper:")
    print(f"   ✓ Bio-cybernetic loop (physiology ↔ policy ↔ feedback)")
    print(f"   ✓ POMDP formulation (state + observation)")
    print(f"   ✓ Multi-objective reward (speed + safety + cognitive load)")
    print(f"   ✓ Non-learnable safety gating (firmware-level constraint)")
    print(f"   ✓ Physiological integration (ECG, RMSSD, stress)")
    
    print("\n" + "=" * 70)
    print("🏁 Execution complete! Thank you for using Bio-Adaptive PoC")
    print("=" * 70 + "\n")
    
    env.close()
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
