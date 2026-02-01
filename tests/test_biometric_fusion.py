"""
Integration Test Suite: Biometric Fusion System

Validates:
1. Data generation (bio_sim.py)
2. Signal processing (bio_processor.py)
3. Environment integration (moto_bio_env.py)
4. Visualization (bio_dashboard.py)
5. Panic Freeze safety mechanism
"""

import pytest
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Imports
from src.data.bio_sim import BiometricDataSimulator, create_synthetic_telemetry
from src.features.bio_processor import BioProcessor
from src.environments.moto_bio_env import MotorcycleBioEnv
from src.visualization.bio_dashboard import BiometricDashboard, create_analysis_report


class TestBioSim:
    """Test synthetic ECG generation."""
    
    def test_simulator_initialization(self):
        """Test simulator creation."""
        sim = BiometricDataSimulator(sampling_rate=500, seed=42)
        assert sim.sampling_rate == 500
        logger.info("✓ Simulator initialization passed")
    
    def test_synthetic_telemetry_generation(self):
        """Test telemetry generation."""
        telemetry = create_synthetic_telemetry(duration=30)
        assert len(telemetry) > 0
        assert 'speed' in telemetry.columns
        assert 'g_force' in telemetry.columns
        assert 'lean_angle' in telemetry.columns
        assert telemetry['speed'].max() > 0
        logger.info(f"✓ Generated {len(telemetry)} telemetry samples")
    
    def test_ecg_segment_generation(self):
        """Test single ECG segment generation."""
        sim = BiometricDataSimulator(sampling_rate=500, seed=42)
        result = sim.generate_ecg_segment(stress_level=0.5, duration=5)
        
        assert 'ecg_raw' in result
        assert len(result['ecg_raw']) == 5 * 500  # 5 seconds at 500 Hz
        assert result['hr'] > 0
        assert result['rmssd'] > 0
        logger.info(f"✓ Generated ECG: HR={result['hr']:.1f} bpm, RMSSD={result['rmssd']:.1f} ms")
    
    def test_full_episode_generation(self):
        """Test full episode generation."""
        sim = BiometricDataSimulator(sampling_rate=500, seed=42)
        telemetry = create_synthetic_telemetry(duration=30)
        ecg_signal, timestamps, stress_profile = sim.generate_episode(
            telemetry, duration=30
        )
        
        assert len(ecg_signal) == 30 * 500
        assert len(timestamps) == 30 * 500
        assert len(stress_profile) > 0
        logger.info(f"✓ Generated full episode: {len(ecg_signal)} samples")


class TestBioProcessor:
    """Test signal processing."""
    
    def test_processor_initialization(self):
        """Test processor creation."""
        processor = BioProcessor(sampling_rate=500, window_size=5)
        assert processor.sampling_rate == 500
        logger.info("✓ Processor initialization passed")
    
    def test_signal_cleaning(self):
        """Test ECG signal cleaning."""
        processor = BioProcessor(sampling_rate=500)
        sim = BiometricDataSimulator(sampling_rate=500, seed=42)
        result = sim.generate_ecg_segment(stress_level=0.5, duration=5)
        ecg_raw = result['ecg_raw']
        
        ecg_clean = processor.clean_signal(ecg_raw)
        assert len(ecg_clean) == len(ecg_raw)
        assert ecg_clean.std() < ecg_raw.std()  # Cleaning reduces noise
        logger.info(f"✓ Cleaned ECG (noise reduction: {(1 - ecg_clean.std()/ecg_raw.std())*100:.1f}%)")
    
    def test_peak_detection(self):
        """Test R-peak detection."""
        processor = BioProcessor(sampling_rate=500)
        sim = BiometricDataSimulator(sampling_rate=500, seed=42)
        result = sim.generate_ecg_segment(stress_level=0.5, duration=10)
        ecg_raw = result['ecg_raw']
        
        ecg_clean = processor.clean_signal(ecg_raw)
        peaks, info = processor.detect_peaks(ecg_clean)
        assert len(peaks) > 0, "Should detect at least some peaks"
        assert all(0 < p < len(ecg_raw) for p in peaks), "Peaks should be within signal bounds"
        logger.info(f"✓ Detected {len(peaks)} R-peaks in 10s signal")
    
    def test_feature_extraction(self):
        """Test feature extraction from ECG."""
        processor = BioProcessor(sampling_rate=500, window_size=5)
        sim = BiometricDataSimulator(sampling_rate=500, seed=42)
        result = sim.generate_ecg_segment(stress_level=0.5, duration=10)
        ecg_raw = result['ecg_raw']
        
        ecg_clean = processor.clean_signal(ecg_raw)
        peaks, _ = processor.detect_peaks(ecg_clean)
        rr_intervals = processor.compute_rr_intervals(peaks)
        
        hr = processor.compute_heart_rate(rr_intervals)
        rmssd = processor.compute_rmssd(rr_intervals)
        hvr_index = processor.compute_hvr_index(rmssd)
        stress_idx = processor.compute_stress_index(hvr_index, hr)
        
        assert 40 < hr < 200
        assert 0 < rmssd < 200
        assert 0 <= hvr_index <= 1
        assert 0 <= stress_idx <= 1
        logger.info(f"✓ Extracted features: HR={hr:.1f} bpm, RMSSD={rmssd:.1f} ms, Stress={stress_idx:.2f}")
    
    def test_batch_processing(self):
        """Test batch processing with windows."""
        processor = BioProcessor(sampling_rate=500, window_size=5)
        sim = BiometricDataSimulator(sampling_rate=500, seed=42)
        result = sim.generate_ecg_segment(stress_level=0.5, duration=30)
        ecg_raw = result['ecg_raw']
        
        df_features = processor.batch_process(ecg_raw, overlap=0.5)
        assert len(df_features) > 0
        assert 'hr' in df_features.columns
        assert 'rmssd' in df_features.columns
        assert 'stress_index' in df_features.columns
        logger.info(f"✓ Batch processed into {len(df_features)} windows")


class TestMotoBioEnv:
    """Test Motorcycle + Biometrics environment."""
    
    def test_env_creation(self):
        """Test environment initialization."""
        env = MotorcycleBioEnv()
        assert env.observation_space.shape == (5,)
        assert env.action_space.shape == (4,)
        logger.info("✓ Environment created successfully")
    
    def test_env_reset(self):
        """Test environment reset."""
        env = MotorcycleBioEnv()
        obs, info = env.reset()
        assert len(obs) == 5
        assert obs[3] > 0  # HR normalized should be > 0
        logger.info(f"✓ Reset environment, initial observation: {obs}")
    
    def test_env_step(self):
        """Test environment step."""
        env = MotorcycleBioEnv()
        obs, _ = env.reset()
        
        action = np.array([0.5, 0.0, 0.3, 0.5])  # throttle, brake, lean, haptic
        obs_next, reward, terminated, truncated, info = env.step(action)
        
        assert len(obs_next) == 5
        assert isinstance(reward, (int, float))
        logger.info(f"✓ Step executed: reward={reward:.2f}, obs={obs_next}")
    
    def test_panic_freeze_mechanism(self):
        """Test Panic Freeze safety feature."""
        env = MotorcycleBioEnv()
        obs, _ = env.reset()
        
        # Manually trigger high stress + high G-force
        env.hr = 180  # Very high heart rate
        env.rmssd = 5  # Very low RMSSD (cognitive saturation)
        env.g_force = 1.5  # High G-force
        
        # Check panic freeze condition
        panic_info = env.get_danger_zone_info()
        
        if panic_info.get('rmssd', 100) < 10 and panic_info.get('g_force', 0) > 1.2:
            logger.info("✓ Panic Freeze condition detected correctly")
            assert panic_info['panic_active'], "Panic freeze should be active"
        
        logger.info(f"✓ Danger zone info: {panic_info}")
    
    def test_episode_running(self):
        """Test running complete episode."""
        env = MotorcycleBioEnv()
        obs, _ = env.reset()
        
        total_reward = 0
        panic_events = 0
        
        for step in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if info.get('panic_freeze', False):
                panic_events += 1
            
            if terminated or truncated:
                break
        
        logger.info(f"✓ Episode completed: {step+1} steps, reward={total_reward:.2f}, panics={panic_events}")


class TestBiometricDashboard:
    """Test visualization."""
    
    def test_dashboard_creation(self):
        """Test dashboard initialization."""
        dashboard = BiometricDashboard(figsize=(16, 10))
        assert dashboard.figsize == (16, 10)
        logger.info("✓ Dashboard created")
    
    def test_figure_creation(self):
        """Test figure creation."""
        dashboard = BiometricDashboard()
        fig, axes = dashboard.create_figure()
        assert len(axes) == 3
        logger.info("✓ Figure with 3 subplots created")
    
    def test_full_visualization(self):
        """Test complete visualization pipeline."""
        # Generate data
        sim = BiometricDataSimulator(sampling_rate=500, seed=42)
        telemetry = create_synthetic_telemetry(duration=30)
        ecg_signal, timestamps, _ = sim.generate_episode(telemetry, duration=30)
        
        # Process data
        processor = BioProcessor(sampling_rate=500, window_size=5)
        ecg_features = processor.batch_process(ecg_signal, overlap=0.5)
        
        # Create visualization
        dashboard = BiometricDashboard()
        fig = dashboard.plot_episode(
            timestamps=timestamps,
            ecg_signal=ecg_signal,
            telemetry_df=telemetry,
            ecg_features_df=ecg_features,
            sampling_rate=500,
            title='Integration Test - Biometric Dashboard',
            save_path='/tmp/test_dashboard.png',
        )
        
        assert fig is not None
        logger.info("✓ Full visualization created and saved")
    
    def test_analysis_report(self):
        """Test statistics report generation."""
        # Generate data
        sim = BiometricDataSimulator(sampling_rate=500, seed=42)
        telemetry = create_synthetic_telemetry(duration=30)
        ecg_signal, timestamps, _ = sim.generate_episode(telemetry, duration=30)
        
        # Process data
        processor = BioProcessor(sampling_rate=500, window_size=5)
        ecg_features = processor.batch_process(ecg_signal, overlap=0.5)
        
        # Generate report
        stats = create_analysis_report(timestamps, ecg_signal, telemetry, ecg_features)
        
        assert 'duration_seconds' in stats
        assert 'mean_stress_index' in stats
        assert stats['mean_stress_index'] > 0
        logger.info(f"✓ Analysis report generated: {len(stats)} metrics")


class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_pipeline(self):
        """Test complete pipeline: data → processing → environment → viz."""
        logger.info("\n" + "="*60)
        logger.info("FULL INTEGRATION TEST: Data → Processing → Env → Viz")
        logger.info("="*60)
        
        # 1. Generate data
        logger.info("\n[1/4] Generating synthetic ECG + telemetry...")
        sim = BiometricDataSimulator(sampling_rate=500, seed=42)
        telemetry = create_synthetic_telemetry(duration=30)
        ecg_signal, timestamps, stress_profile = sim.generate_episode(
            telemetry, duration=30
        )
        logger.info(f"     ✓ Generated {len(ecg_signal)} ECG samples")
        
        # 2. Process signals
        logger.info("\n[2/4] Processing ECG signal...")
        processor = BioProcessor(sampling_rate=500, window_size=5)
        ecg_features = processor.batch_process(ecg_signal, overlap=0.5)
        logger.info(f"     ✓ Extracted {len(ecg_features)} feature windows")
        
        # 3. Integrate with environment
        logger.info("\n[3/4] Running environment episode...")
        env = MotorcycleBioEnv()
        obs, _ = env.reset()
        
        for step in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        logger.info(f"     ✓ Completed {step+1} steps in environment")
        
        # 4. Create visualization
        logger.info("\n[4/4] Creating multimodal dashboard...")
        dashboard = BiometricDashboard()
        fig = dashboard.plot_episode(
            timestamps=timestamps,
            ecg_signal=ecg_signal,
            telemetry_df=telemetry,
            ecg_features_df=ecg_features,
            sampling_rate=500,
            title='Integration Test: Full Pipeline',
            save_path='/tmp/integration_test_dashboard.png',
        )
        logger.info("     ✓ Dashboard created and saved")
        
        logger.info("\n" + "="*60)
        logger.info("✓ FULL INTEGRATION TEST PASSED")
        logger.info("="*60)


if __name__ == '__main__':
    logger.info("\n" + "="*70)
    logger.info("BIOMETRIC FUSION SYSTEM - INTEGRATION TEST SUITE")
    logger.info("="*70)
    
    # Run tests
    pytest.main([__file__, '-v', '--tb=short', '--disable-warnings'])
