"""
Unit tests for biometric signal generation and correlation.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from generate_biometrics import (
    generate_ecg_signal,
    correlate_biometrics_with_telemetry,
    generate_biometric_dataset,
    save_biometric_data,
)


class TestECGGeneration:
    """Test ECG signal synthesis with physiological morphology."""

    def test_ecg_length(self):
        """Validate ECG array has correct length."""
        n_samples = 500
        ecg, hr, hrv = generate_ecg_signal(n_samples, sampling_rate=500, base_hr=70)
        
        assert len(ecg) == n_samples
        assert len(hr) == n_samples
        assert len(hrv) == n_samples

    def test_hr_stress_modulation(self):
        """Validate HR increases with stress."""
        n_samples = 1000
        stress_low = np.zeros(n_samples)
        stress_high = np.ones(n_samples)
        
        _, hr_low, _ = generate_ecg_signal(n_samples, sampling_rate=500, base_hr=70, stress_modulation=stress_low)
        _, hr_high, _ = generate_ecg_signal(n_samples, sampling_rate=500, base_hr=70, stress_modulation=stress_high)
        
        assert np.mean(hr_high) > np.mean(hr_low)

    def test_hrv_stress_modulation(self):
        """Validate HRV decreases with stress."""
        n_samples = 1000
        stress_low = np.zeros(n_samples)
        stress_high = np.ones(n_samples)
        
        _, _, hrv_low = generate_ecg_signal(n_samples, sampling_rate=500, base_hr=70, stress_modulation=stress_low)
        _, _, hrv_high = generate_ecg_signal(n_samples, sampling_rate=500, base_hr=70, stress_modulation=stress_high)
        
        assert np.mean(hrv_low) > np.mean(hrv_high)

    def test_ecg_no_nan(self):
        """Ensure ECG has no NaN values."""
        n_samples = 500
        ecg, hr, hrv = generate_ecg_signal(n_samples, sampling_rate=500, base_hr=70)
        
        assert not np.isnan(ecg).any()
        assert not np.isnan(hr).any()
        assert not np.isnan(hrv).any()


class TestStressCorrelation:
    """Test stress modulation from telemetry."""

    def test_stress_from_brake_signal(self):
        """Verify stress correlates with brake signal."""
        n_samples = 500
        n_bio_samples = 100
        
        telemetry = np.zeros((n_samples, 8))
        telemetry[100:200, 5] = 0.8  # Brake pressure
        
        stress = correlate_biometrics_with_telemetry(telemetry, n_bio_samples)
        
        assert len(stress) == n_bio_samples
        assert 0.0 <= stress.min() and stress.max() <= 1.0

    def test_stress_array_bounds(self):
        """Verify stress stays in [0, 1]."""
        n_samples = 300
        n_bio_samples = 60
        
        telemetry = np.random.randn(n_samples, 8) * 0.1
        telemetry[:, 5] = np.random.uniform(0, 1, n_samples)
        
        stress = correlate_biometrics_with_telemetry(telemetry, n_bio_samples)
        
        assert stress.min() >= 0.0 and stress.max() <= 1.0


class TestBiometricDataset:
    """Test complete biometric dataset generation."""

    def test_generate_biometric_dataset_shape(self):
        """Validate dataset generation produces correct shapes."""
        n_samples = 100
        timestamps, ecg, hr, hrv, stress = generate_biometric_dataset(
            n_samples=n_samples,
            sampling_rate=50,
            ecg_sampling_rate=500
        )
        
        expected_ecg_samples = n_samples * 500 // 50
        assert len(ecg) == expected_ecg_samples
        assert len(hr) == expected_ecg_samples
        assert len(hrv) == expected_ecg_samples
        assert len(stress) == expected_ecg_samples

    def test_full_pipeline_valid(self):
        """Validate complete generation pipeline."""
        n_samples = 50
        timestamps, ecg, hr, hrv, stress = generate_biometric_dataset(
            n_samples=n_samples,
            sampling_rate=50,
            ecg_sampling_rate=500
        )
        
        assert len(ecg) == len(hr) == len(hrv) == len(stress) == len(timestamps)
        assert not np.isnan(ecg).any()
        assert not np.isnan(hr).any()


class TestBiometricDataSaving:
    """Test CSV output and data persistence."""

    def test_save_and_load_biometric_data(self):
        """Verify CSV output format and integrity."""
        n_samples = 50
        timestamps, ecg, hr, hrv, stress = generate_biometric_dataset(
            n_samples=n_samples, sampling_rate=50, ecg_sampling_rate=500
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "biometrics.csv"
            save_biometric_data(Path(output_path), timestamps, ecg, hr, hrv, stress)
            
            assert output_path.exists()
            df = pd.read_csv(output_path)
            assert len(df) == len(ecg)
            assert set(df.columns) == {"timestamp", "ecg", "heart_rate", "hrv", "stress_level"}

    def test_csv_data_ranges(self):
        """Validate saved data is in physiological ranges."""
        n_samples = 50
        timestamps, ecg, hr, hrv, stress = generate_biometric_dataset(
            n_samples=n_samples, sampling_rate=50, ecg_sampling_rate=500
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "biometrics.csv"
            save_biometric_data(Path(output_path), timestamps, ecg, hr, hrv, stress)
            
            df = pd.read_csv(output_path)
            
            # HR: 50-150 bpm
            assert df["heart_rate"].min() >= 50
            assert df["heart_rate"].max() <= 150
            
            # Stress: 0-1
            assert df["stress_level"].min() >= 0
            assert df["stress_level"].max() <= 1.0

    def test_csv_no_nan(self):
        """Ensure saved CSV has no NaN values."""
        n_samples = 50
        timestamps, ecg, hr, hrv, stress = generate_biometric_dataset(
            n_samples=n_samples, sampling_rate=50, ecg_sampling_rate=500
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "biometrics.csv"
            save_biometric_data(Path(output_path), timestamps, ecg, hr, hrv, stress)
            
            df = pd.read_csv(output_path)
            assert df.isna().sum().sum() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
