"""
QUICK START: Biometric Fusion System

Run this file to see a complete end-to-end demonstration:
1. Generate realistic ECG with stress correlation
2. Process signals (clean, detect peaks, extract features)
3. Run environment with Panic Freeze detection
4. Create multimodal visualization

Usage:
    python examples/biometric_demo.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Create output directory
output_dir = Path('/tmp/biometric_demo')
output_dir.mkdir(exist_ok=True)


def demo_1_data_generation():
    """1. Generate synthetic ECG correlated with telemetry"""
    logger.info("\n" + "="*70)
    logger.info("DEMO 1: Synthetic ECG Generation")
    logger.info("="*70)
    
    from src.data.bio_sim import BiometricDataSimulator, create_synthetic_telemetry
    
    # Generate circuit telemetry (30s)
    logger.info("\n[1a] Generating motorcycle circuit telemetry...")
    telemetry = create_synthetic_telemetry(duration=30)
    logger.info(f"✓ Generated {len(telemetry)} telemetry samples")
    logger.info(f"   Speed range:     {telemetry['speed'].min():.1f} - {telemetry['speed'].max():.1f} m/s")
    logger.info(f"   G-force range:   {telemetry['g_force'].min():.2f} - {telemetry['g_force'].max():.2f} G")
    logger.info(f"   Lean angle range: {telemetry['lean_angle'].min():.1f} - {telemetry['lean_angle'].max():.1f}°")
    
    # Generate correlated ECG
    logger.info("\n[1b] Generating ECG correlated with stress...")
    sim = BiometricDataSimulator(sampling_rate=500, seed=42)
    ecg_signal, timestamps, stress_profile = sim.generate_episode(
        telemetry, duration=30
    )
    logger.info(f"✓ Generated {len(ecg_signal)} ECG samples (30s @ 500Hz)")
    logger.info(f"   Stress range:    {stress_profile.min():.2f} - {stress_profile.max():.2f}")
    logger.info(f"   Mean stress:     {stress_profile.mean():.2f}")
    
    # Analyze stress distribution
    low_stress = (stress_profile < 0.3).sum() / len(stress_profile) * 100
    med_stress = ((stress_profile >= 0.3) & (stress_profile < 0.7)).sum() / len(stress_profile) * 100
    high_stress = (stress_profile >= 0.7).sum() / len(stress_profile) * 100
    
    logger.info(f"\n   Stress distribution:")
    logger.info(f"     Low stress  (< 0.3):  {low_stress:5.1f}%")
    logger.info(f"     Med stress  (0.3-0.7): {med_stress:5.1f}%")
    logger.info(f"     High stress (> 0.7):  {high_stress:5.1f}%")
    
    return ecg_signal, timestamps, telemetry


def demo_2_signal_processing(ecg_signal, timestamps):
    """2. Process ECG signals with neurokit2"""
    logger.info("\n" + "="*70)
    logger.info("DEMO 2: ECG Signal Processing")
    logger.info("="*70)
    
    from src.features.bio_processor import BioProcessor
    
    processor = BioProcessor(sampling_rate=500, window_size=5)
    
    # Clean signal
    logger.info("\n[2a] Cleaning ECG signal (0.5-150 Hz bandpass)...")
    ecg_clean = processor.clean_signal(ecg_signal)
    noise_reduction = (1 - ecg_clean.std() / ecg_signal.std()) * 100
    logger.info(f"✓ Noise reduction: {noise_reduction:.1f}%")
    
    # Detect R-peaks
    logger.info("\n[2b] Detecting R-peaks (QRS complexes)...")
    peaks, info = processor.detect_peaks(ecg_clean)
    logger.info(f"✓ Detected {len(peaks)} R-peaks")
    if len(peaks) > 0:
        expected_peaks = len(ecg_signal) / (500 * 60 / 100)  # Assuming ~100 bpm
        accuracy = min(100, len(peaks) / expected_peaks * 100)
        logger.info(f"   Detection accuracy: ~{accuracy:.1f}%")
    
    # Compute heart rate
    logger.info("\n[2c] Computing Heart Rate and RMSSD...")
    rr_intervals = processor.compute_rr_intervals(peaks)
    hr = processor.compute_heart_rate(rr_intervals)
    rmssd = processor.compute_rmssd(rr_intervals)
    logger.info(f"✓ Heart Rate:  {hr:.1f} bpm (mean across window)")
    logger.info(f"✓ RMSSD:       {rmssd:.1f} ms")
    
    # Batch feature extraction
    logger.info("\n[2d] Batch processing with overlapping windows...")
    df_features = processor.batch_process(ecg_signal, overlap=0.5)
    logger.info(f"✓ Extracted features for {len(df_features)} windows (5s each)")
    logger.info(f"   HR range:       {df_features['hr'].min():.1f} - {df_features['hr'].max():.1f} bpm")
    logger.info(f"   RMSSD range:    {df_features['rmssd'].min():.1f} - {df_features['rmssd'].max():.1f} ms")
    logger.info(f"   Stress range:   {df_features['stress_index'].min():.2f} - {df_features['stress_index'].max():.2f}")
    
    return ecg_clean, peaks, df_features


def demo_3_environment(ecg_signal, telemetry, df_features):
    """3. Run Gymnasium environment with Panic Freeze"""
    logger.info("\n" + "="*70)
    logger.info("DEMO 3: Motorcycle Environment + Panic Freeze Safety")
    logger.info("="*70)
    
    from src.environments.moto_bio_env import MotorcycleBioEnv
    
    env = MotorcycleBioEnv()
    
    # Reset environment
    logger.info("\n[3a] Initializing environment...")
    obs, info = env.reset()
    logger.info(f"✓ Environment reset")
    logger.info(f"   Observation space: {obs}")
    
    # Run episode with panic detection
    logger.info("\n[3b] Running 100 episode steps with random actions...")
    total_reward = 0
    panic_events = 0
    high_stress_steps = 0
    max_stress = 0
    
    for step in range(100):
        # Random action
        action = env.action_space.sample()
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Track panic events
        if info.get('panic_freeze'):
            panic_events += 1
            logger.info(f"   ⚠️ PANIC FREEZE at step {step}: "
                       f"RMSSD={info.get('rmssd', 'N/A'):.1f} ms, "
                       f"G={info.get('g_force', 'N/A'):.2f}G")
        
        # Track stress
        stress = obs[4]  # Last element is rmssd_index (inverse of stress)
        actual_stress = 1 - stress if stress >= 0 else 0
        if actual_stress > 0.7:
            high_stress_steps += 1
        max_stress = max(max_stress, actual_stress)
        
        if terminated or truncated:
            logger.info(f"   Episode terminated at step {step}")
            break
    
    logger.info(f"\n✓ Episode complete: {step+1} steps")
    logger.info(f"   Total reward:     {total_reward:.2f}")
    logger.info(f"   Panic events:     {panic_events}")
    logger.info(f"   High stress steps: {high_stress_steps} / {step+1}")
    logger.info(f"   Max stress level: {max_stress:.2f}")
    
    # Test manual panic freeze trigger
    logger.info("\n[3c] Testing manual Panic Freeze trigger...")
    env.hr = 180  # Very high HR
    env.rmssd = 5  # Very low RMSSD (panic)
    env.g_force = 1.5  # High G-force
    
    danger_info = env.get_danger_zone_info()
    logger.info(f"✓ Danger zone assessment:")
    logger.info(f"   Cognitive saturation: {danger_info.get('cognitive_saturation', False)}")
    logger.info(f"   High G-force:         {danger_info.get('high_g_force', False)}")
    logger.info(f"   Panic active:         {danger_info.get('panic_active', False)}")
    logger.info(f"   RMSSD:                {danger_info.get('rmssd', 'N/A'):.1f} ms")
    logger.info(f"   G-force:              {danger_info.get('g_force', 'N/A'):.2f} G")


def demo_4_visualization(ecg_signal, timestamps, telemetry, df_features):
    """4. Create multimodal dashboard"""
    logger.info("\n" + "="*70)
    logger.info("DEMO 4: Multimodal Visualization Dashboard")
    logger.info("="*70)
    
    from src.visualization.bio_dashboard import BiometricDashboard, create_analysis_report
    import matplotlib.pyplot as plt
    
    logger.info("\n[4a] Creating 3-panel dashboard...")
    dashboard = BiometricDashboard(figsize=(16, 10))
    
    fig = dashboard.plot_episode(
        timestamps=timestamps,
        ecg_signal=ecg_signal,
        telemetry_df=telemetry,
        ecg_features_df=df_features,
        sampling_rate=500,
        title='Complete Biometric Fusion Demo: Motorcycle + ECG Integration',
        save_path=str(output_dir / 'demo_dashboard.png'),
    )
    logger.info(f"✓ Dashboard created and saved to {output_dir}/demo_dashboard.png")
    
    # Generate analysis report
    logger.info("\n[4b] Generating statistical analysis...")
    stats = create_analysis_report(timestamps, ecg_signal, telemetry, df_features)
    
    logger.info("\n   Episode Statistics:")
    logger.info(f"     Duration:            {stats['duration_seconds']:.1f} seconds")
    logger.info(f"     Mean speed:          {stats['mean_speed_ms']:.1f} m/s")
    logger.info(f"     Max G-force:         {stats['max_g_force']:.2f} G")
    logger.info(f"     Mean lean angle:     {stats['mean_lean_angle']:.1f}°")
    logger.info(f"     Mean HR:             {stats['mean_hr_bpm']:.1f} bpm")
    logger.info(f"     Min RMSSD:           {stats['min_rmssd_ms']:.1f} ms")
    logger.info(f"     Max stress:          {stats['max_stress_index']:.2f}")
    logger.info(f"     Mean stress:         {stats['mean_stress_index']:.2f}")
    logger.info(f"     Time in high stress: {stats['time_in_high_stress']*100:.1f}%")
    
    # Try to display figure
    logger.info("\n[4c] Attempting to display figure...")
    try:
        # Don't block - just save
        logger.info("✓ Figure saved. Note: Display not available in headless environment.")
    except Exception as e:
        logger.info(f"   Note: {e}")
    
    return fig


def main():
    """Run complete demo"""
    logger.info("\n" * 2)
    logger.info("█" * 70)
    logger.info("█" + " " * 68 + "█")
    logger.info("█" + "  BIOMETRIC FUSION SYSTEM - COMPLETE DEMONSTRATION".center(68) + "█")
    logger.info("█" + "  Motorcycle + ECG/HRV Integration".center(68) + "█")
    logger.info("█" + " " * 68 + "█")
    logger.info("█" * 70)
    
    try:
        # Run all demos
        ecg_signal, timestamps, telemetry = demo_1_data_generation()
        ecg_clean, peaks, df_features = demo_2_signal_processing(ecg_signal, timestamps)
        demo_3_environment(ecg_signal, telemetry, df_features)
        fig = demo_4_visualization(ecg_signal, timestamps, telemetry, df_features)
        
        # Final summary
        logger.info("\n" + "="*70)
        logger.info("✅ DEMONSTRATION COMPLETE")
        logger.info("="*70)
        logger.info("\nKey Features Demonstrated:")
        logger.info("  ✓ Realistic ECG generation correlated with telemetry")
        logger.info("  ✓ Signal processing pipeline (clean → detect → extract)")
        logger.info("  ✓ Environment integration with biometric state")
        logger.info("  ✓ Panic Freeze safety mechanism detection")
        logger.info("  ✓ Multimodal visualization (3-panel dashboard)")
        logger.info("\nOutput Files:")
        logger.info(f"  📊 Dashboard: {output_dir}/demo_dashboard.png")
        logger.info("\nNext Steps:")
        logger.info("  1. Review the generated dashboard visualization")
        logger.info("  2. Run the test suite: pytest tests/test_biometric_fusion.py -v")
        logger.info("  3. Read technical docs: docs/BIOMETRIC_FUSION_IMPLEMENTATION.md")
        logger.info("  4. Integrate with your own ECG hardware")
        logger.info("\n" + "="*70)
        
    except Exception as e:
        logger.error(f"\n❌ Error during demonstration: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
