#!/usr/bin/env python3
"""
Quick Demo Script - Fast execution for testing
Runs a quick version with reduced parameters for demonstration
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import SIM_CONFIG, TRAIN_CONFIG, PATHS
from src.data_gen import SyntheticTelemetry
from src.environment import MotoBioEnv
from src.train import create_training_environment, train_ppo_agent
from src.visualize import create_evaluation_lap, create_visualization

def main():
    print("\n" + "="*70)
    print("  Bio-Adaptive Haptic Coaching - QUICK DEMO (5 min)")
    print("="*70 + "\n")
    
    print("⚡ Using reduced parameters for quick demonstration...")
    print("   - 10 laps (vs 100)")
    print("   - 10,000 timesteps (vs 100,000)")
    print("   - No checkpoints\n")
    
    # Phase 1: Generate minimal data
    print("📊 Phase 1: Generating data (10 laps)...")
    gen = SyntheticTelemetry()
    session = gen.generate_race_session(n_laps=10)
    print(f"✅ Generated {len(session.telemetry_df)} observations\n")
    
    # Phase 2: Create environment
    print("🎮 Phase 2: Creating environment...")
    env = MotoBioEnv(telemetry_df=session.telemetry_df)
    obs, _ = env.reset()
    print(f"✅ Environment ready: state={obs}\n")
    
    # Phase 3: Quick training
    print("🚀 Phase 3: Training PPO (10k steps)...")
    train_env, _ = create_training_environment(n_laps=5, num_envs=1)
    model, metrics = train_ppo_agent(
        env=train_env,
        total_timesteps=10000,
        save_dir=PATHS.MODELS_DIR
    )
    print(f"✅ Training complete! Mean reward: {metrics['mean_reward']:.2f}\n")
    
    # Phase 4: Evaluate and visualize
    print("📈 Phase 4: Evaluation & Visualization...")
    model_path = str(PATHS.MODELS_DIR / "ppo_bio_adaptive")
    trajectory, ecg = create_evaluation_lap(model_path, env, session.telemetry_df)
    
    print(f"✅ Evaluation complete!")
    create_visualization(trajectory, session.telemetry_df, ecg)
    
    print("\n" + "="*70)
    print("  DEMO COMPLETE! Check logs/ folder for results")
    print("="*70 + "\n")
    
    print("📁 Output files:")
    print(f"   {PATHS.LOGS_DIR / 'bio_adaptive_results.png'}")
    print(f"   {PATHS.LOGS_DIR / 'training_metrics.json'}")
    print("\n💡 For full production run:")
    print("   python scripts/run_pipeline.py\n")

if __name__ == "__main__":
    main()
