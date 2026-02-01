"""
Master Pipeline Orchestrator
Executes full workflow: Data Gen -> Train -> Eval -> Visualize
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import (
    PATHS, SIM_CONFIG, TRAIN_CONFIG, VIS_CONFIG,
    get_config_summary
)
from src.data_gen import SyntheticTelemetry
from src.environment import MotoBioEnv
from src.train import create_training_environment, train_ppo_agent
from src.visualize import (
    create_evaluation_lap, create_visualization,
    visualize_training_metrics, save_evaluation_report
)


def print_header(text: str) -> None:
    """Print formatted header"""
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}\n")


def print_config_summary() -> None:
    """Print configuration summary"""
    print_header("CONFIGURATION SUMMARY")
    summary = get_config_summary()
    for key, value in summary.items():
        print(f"  {key:.<40} {value}")


def phase_1_data_generation() -> Dict[str, Any]:
    """Phase 1: Generate Synthetic Telemetry"""
    print_header("PHASE 1: SYNTHETIC DATA GENERATION")
    
    print(f"Circuit Parameters:")
    print(f"  Circuit Length ........ {SIM_CONFIG.CIRCUIT_LENGTH_KM} km")
    print(f"  Number of Laps ....... {SIM_CONFIG.NUM_LAPS}")
    print(f"  Sampling Rate ........ {SIM_CONFIG.SAMPLING_RATE_TELEMETRY} Hz")
    
    print(f"\nPhysiological Parameters:")
    print(f"  Resting HR ........... {SIM_CONFIG.RESTING_HR} bpm")
    print(f"  Max HR ............... {SIM_CONFIG.MAX_HR} bpm")
    print(f"  Panic Threshold ...... {SIM_CONFIG.PANIC_THRESHOLD}")
    
    print(f"\n⏳ Generating data...")
    gen = SyntheticTelemetry()
    session = gen.generate_race_session(n_laps=SIM_CONFIG.NUM_LAPS)
    
    print(f"\n📊 Data Generation Complete:")
    print(f"  Total Samples ........ {len(session.telemetry_df)}")
    print(f"  Duration ............ {session.metadata['total_duration_seconds']:.1f}s")
    print(f"  Mean Speed .......... {session.metadata['mean_speed_kmh']:.1f} km/h")
    print(f"  Max Speed ........... {session.metadata['max_speed_kmh']:.1f} km/h")
    print(f"  Mean HR ............ {session.metadata['mean_hr_bpm']:.1f} bpm")
    
    return {
        "telemetry_df": session.telemetry_df,
        "ecg_signal": session.ecg_signal,
        "hrv_metrics": session.hrv_metrics,
        "metadata": session.metadata,
    }


def phase_2_environment_setup(data: Dict) -> MotoBioEnv:
    """Phase 2: Create Bio-Adaptive Environment"""
    print_header("PHASE 2: ENVIRONMENT SETUP")
    
    print(f"Creating MotoBioEnv with Bio-Gating:")
    print(f"  State Space ......... 5D [Speed, Lean, G-Force, HRV, Stress]")
    print(f"  Action Space ....... 4 [No Feedback, Mild, Warning, Emergency]")
    print(f"  Bio-Gate Rule ...... IF stress > {SIM_CONFIG.PANIC_THRESHOLD}, force action=0")
    
    print(f"\nReward Function:")
    print(f"  Speed Weight ........ 0.50")
    print(f"  Safety Weight ....... 0.35")
    print(f"  Stress Penalty ...... -0.15 × stress²")
    
    env = MotoBioEnv(telemetry_df=data["telemetry_df"])
    
    print(f"\n✅ Environment Created:")
    print(f"  Observation Space ... {env.observation_space}")
    print(f"  Action Space ........ {env.action_space}")
    
    return env


def phase_3_rl_training(data: Dict) -> Dict[str, Any]:
    """Phase 3: Train PPO Agent"""
    print_header("PHASE 3: RL TRAINING (PPO)")
    
    print(f"Training Configuration:")
    print(f"  Total Timesteps ...... {TRAIN_CONFIG.TOTAL_TIMESTEPS}")
    print(f"  Learning Rate ....... {TRAIN_CONFIG.LEARNING_RATE}")
    print(f"  N Steps ............ {TRAIN_CONFIG.N_STEPS}")
    print(f"  Gamma (Discount) ... {TRAIN_CONFIG.GAMMA}")
    print(f"  Network Layers ..... {TRAIN_CONFIG.POLICY_NETWORK_LAYERS}")
    
    # Create training environment
    env, _ = create_training_environment(
        n_laps=min(50, SIM_CONFIG.NUM_LAPS),
        num_envs=1
    )
    
    # Train agent
    model, train_metrics = train_ppo_agent(
        env=env,
        total_timesteps=TRAIN_CONFIG.TOTAL_TIMESTEPS,
        save_dir=PATHS.MODELS_DIR
    )
    
    print(f"\n✅ Training Complete:")
    print(f"  Mean Reward ........ {train_metrics.get('mean_reward', 0.0):.2f}")
    print(f"  Max Reward ......... {train_metrics.get('max_reward', 0.0):.2f}")
    
    return {
        "model": model,
        "metrics": train_metrics,
    }


def phase_4_visualization_and_eval(data: Dict, train_result: Dict) -> None:
    """Phase 4: Evaluation and Visualization"""
    print_header("PHASE 4: EVALUATION & VISUALIZATION")
    
    # Create evaluation environment
    eval_env = MotoBioEnv(telemetry_df=data["telemetry_df"])
    
    # Run evaluation lap
    model_path = str(PATHS.MODELS_DIR / "ppo_bio_adaptive")
    print(f"⏳ Running evaluation lap...")
    trajectory, ecg = create_evaluation_lap(model_path, eval_env, data["telemetry_df"])
    
    print(f"\n✅ Evaluation Complete:")
    print(f"  Total Steps ......... {len(trajectory['time'])}")
    print(f"  Mean Speed ......... {trajectory['speed'].mean():.1f} km/h")
    print(f"  Max Stress ......... {trajectory['stress'].max():.2f}")
    bio_gates = sum(trajectory['bio_gated'])
    print(f"  Bio-Gate Activations {bio_gates}")
    
    # Create visualization
    print(f"\n⏳ Creating 3-panel visualization...")
    create_visualization(
        trajectory=trajectory,
        telemetry_df=data["telemetry_df"],
        ecg_signal=data["ecg_signal"],
        output_path=PATHS.LOGS_DIR / "bio_adaptive_results.png"
    )
    
    # Save metrics
    print(f"⏳ Saving evaluation report...")
    save_evaluation_report(
        trajectory=trajectory,
        metrics=train_result["metrics"],
        output_path=PATHS.LOGS_DIR / "evaluation_metrics.json"
    )
    
    # Visualization of training metrics
    visualize_training_metrics(
        metrics=train_result["metrics"],
        output_path=PATHS.LOGS_DIR / "training_metrics_plot.png"
    )


def print_summary() -> None:
    """Print final summary"""
    print_header("PIPELINE COMPLETE ✅")
    
    print(f"📁 Output Files:")
    print(f"\n  Data:")
    print(f"    {PATHS.DATA_DIR / 'telemetry.csv'}")
    print(f"    {PATHS.DATA_DIR / 'ecg_signal.npy'}")
    print(f"    {PATHS.DATA_DIR / 'hrv_metrics.json'}")
    
    print(f"\n  Models:")
    print(f"    {PATHS.MODELS_DIR / 'ppo_bio_adaptive.zip'}")
    
    print(f"\n  Results:")
    print(f"    {PATHS.LOGS_DIR / 'bio_adaptive_results.png'}")
    print(f"    {PATHS.LOGS_DIR / 'training_metrics_plot.png'}")
    print(f"    {PATHS.LOGS_DIR / 'evaluation_metrics.json'}")
    print(f"    {PATHS.LOGS_DIR / 'training_metrics.json'}")
    
    print(f"\n⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n🚀 Ready for deployment!")


def main() -> None:
    """Execute full pipeline"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  Bio-Adaptive Haptic Coaching System - Full Pipeline".center(68) + "║")
    print("║" + "  End-to-End MLOps Implementation".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    
    # Print configuration
    print_config_summary()
    
    # Create directories
    PATHS.DATA_DIR.mkdir(parents=True, exist_ok=True)
    PATHS.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PATHS.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # Execute phases
        data = phase_1_data_generation()
        env = phase_2_environment_setup(data)
        train_result = phase_3_rl_training(data)
        phase_4_visualization_and_eval(data, train_result)
        
        # Summary
        print_summary()
        
        return 0
    
    except Exception as e:
        print_header("ERROR OCCURRED")
        print(f"❌ {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
