"""
Synthetic Biometric Data Generation for Multimodal Fusion.

Generates ECG (electrocardiogram), HR (heart rate), HRV (heart rate variability),
and stress levels synchronized with motorcycle telemetry.

Physiological Rules:
- During hard braking: HR increases (sympathetic activation), HRV decreases
- During smooth cornering: HR stable, HRV higher
- Stress = 1 - (HRV / HRV_max), normalized to [0, 1]

Usage:
    python scripts/generate_biometrics.py \
        --telemetry data/processed/pilot_1.csv \
        --output data/processed/pilot_1_biometrics.csv \
        --ecg-frequency 500 \
        --duration 300
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_ecg_signal(
    n_samples: int,
    sampling_rate: int = 500,
    base_hr: float = 70.0,
    stress_modulation: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic ECG signal with HR and HRV modulation.

    Args:
        n_samples: Number of samples
        sampling_rate: ECG sampling rate (Hz)
        base_hr: Baseline heart rate (bpm)
        stress_modulation: Array [0, 1] where 1 = max stress. HR increases, HRV decreases.

    Returns:
        (ecg_signal, heart_rate, hrv)
    """
    if stress_modulation is None:
        stress_modulation = np.zeros(n_samples)

    t = np.arange(n_samples) / sampling_rate

    # Baseline HR modulation from stress
    # At stress=0: HR = base_hr; at stress=1: HR increases by 50 bpm
    hr_array = base_hr + stress_modulation * 50.0

    # HRV (Heart Rate Variability) inversely related to stress
    # At stress=0: HRV = 50 ms; at stress=1: HRV = 5 ms (sympathetic dominance)
    hrv_array = 50.0 - stress_modulation * 45.0
    hrv_array = np.maximum(hrv_array, 5.0)  # Floor at 5 ms

    # Generate synthetic ECG using a simple model
    # R-wave spacing corresponds to HR
    ecg_signal = np.zeros(n_samples)

    # Time-varying R-R intervals (ms)
    rr_intervals = 60000.0 / hr_array  # Convert bpm to ms

    # Add HRV to RR intervals
    rr_intervals_with_hrv = rr_intervals + np.random.randn(n_samples) * (hrv_array / 10.0)
    rr_intervals_with_hrv = np.maximum(rr_intervals_with_hrv, 300.0)  # Physiological bounds

    # Build ECG as a series of QRS complexes
    sample_idx = 0
    phase = 0.0
    for i in range(n_samples):
        rr_ms = rr_intervals_with_hrv[i]
        rr_samples = int(rr_ms * sampling_rate / 1000.0)

        # Simulate a QRS complex as a triangular wave
        if i % max(rr_samples, 1) == 0:
            phase = 0.0

        # Simple ECG morphology (P-QRS-T)
        if phase < 0.2:  # P wave
            ecg_signal[i] = 0.2 * np.sin(2 * np.pi * phase / 0.2)
        elif phase < 0.3:  # Q wave
            ecg_signal[i] = -0.1 * np.sin(2 * np.pi * (phase - 0.2) / 0.1)
        elif phase < 0.4:  # R wave
            ecg_signal[i] = 1.0 * np.sin(2 * np.pi * (phase - 0.3) / 0.1)
        elif phase < 0.5:  # S wave
            ecg_signal[i] = -0.1 * np.sin(2 * np.pi * (phase - 0.4) / 0.1)
        else:  # T wave
            ecg_signal[i] = 0.2 * np.sin(2 * np.pi * (phase - 0.5) / 0.5)

        phase += 1.0 / (rr_samples + 1)

    # Add physiological noise
    ecg_signal += 0.05 * np.random.randn(n_samples)

    return ecg_signal, hr_array, hrv_array


def correlate_biometrics_with_telemetry(
    telemetry: np.ndarray,
    n_biometric_samples: int,
) -> np.ndarray:
    """Compute stress modulation from telemetry (braking = stress).

    Args:
        telemetry: (T, n_features) array from motorcycle env
                  Column 5 = brake pressure [0, 1]
        n_biometric_samples: Number of biometric samples to generate

    Returns:
        stress_modulation: (n_biometric_samples,) array [0, 1]
    """
    # Downsample/resample telemetry to match biometric sampling
    if len(telemetry) != n_biometric_samples:
        indices = np.linspace(0, len(telemetry) - 1, n_biometric_samples).astype(int)
        brake_signal = telemetry[indices, 5]  # Brake pressure
    else:
        brake_signal = telemetry[:, 5]

    # Stress = f(brake pressure) with some dynamics
    # Hard braking → high stress; coasting → low stress
    stress = brake_signal.copy()

    # Add first-order low-pass filter dynamics (stress doesn't change instantly)
    alpha = 0.1
    for i in range(1, len(stress)):
        stress[i] = alpha * stress[i] + (1 - alpha) * stress[i - 1]

    # Smooth peaks
    stress = np.convolve(stress, np.ones(5) / 5, mode="same")

    return np.clip(stress, 0.0, 1.0)


def generate_biometric_dataset(
    telemetry_path: Path = None,
    n_samples: int = 500,
    sampling_rate: int = 50,
    ecg_sampling_rate: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate complete biometric dataset.

    Returns:
        (timestamps, ecg, hr, hrv, stress_level)
    """
    # Load or generate telemetry
    if telemetry_path and telemetry_path.exists():
        df = pd.read_csv(telemetry_path)
        telemetry = df[["ax", "ay", "az", "gx", "gy", "gz", "throttle", "brake"]].values
        telemetry = telemetry[: int(n_samples * ecg_sampling_rate / sampling_rate)]
    else:
        # Generate synthetic telemetry
        t = np.arange(n_samples) / sampling_rate
        ax = 2.0 * np.sin(0.02 * t)
        ay = 1.5 * np.sin(0.015 * t)
        brake = 0.5 * (1 + np.sin(0.01 * t))  # Periodic braking
        telemetry = np.column_stack([ax, ay, np.zeros(n_samples), np.zeros(n_samples),
                                       np.zeros(n_samples), np.zeros(n_samples),
                                       0.5 * np.ones(n_samples), brake])

    # Compute stress from braking
    n_ecg_samples = int(n_samples * ecg_sampling_rate / sampling_rate)
    stress = correlate_biometrics_with_telemetry(telemetry, n_ecg_samples)

    # Generate ECG with physiological response to stress
    ecg, hr, hrv = generate_ecg_signal(
        n_ecg_samples,
        sampling_rate=ecg_sampling_rate,
        base_hr=70.0,
        stress_modulation=stress,
    )

    # Timestamps for ECG (higher frequency)
    timestamps_ecg = np.arange(n_ecg_samples) / ecg_sampling_rate

    return timestamps_ecg, ecg, hr, hrv, stress


def save_biometric_data(
    output_path: Path,
    timestamps: np.ndarray,
    ecg: np.ndarray,
    hr: np.ndarray,
    hrv: np.ndarray,
    stress: np.ndarray,
) -> None:
    """Save biometric data to CSV."""
    df = pd.DataFrame({
        "timestamp": timestamps,
        "ecg": ecg,
        "heart_rate": hr,
        "hrv": hrv,
        "stress_level": stress,
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Biometric data saved to {output_path}")
    logger.info(f"  Samples: {len(df)}")
    logger.info(f"  HR range: {hr.min():.1f} - {hr.max():.1f} bpm")
    logger.info(f"  HRV range: {hrv.min():.1f} - {hrv.max():.1f} ms")
    logger.info(f"  Stress range: {stress.min():.2f} - {stress.max():.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Synthetic Biometric Data")
    parser.add_argument("--telemetry", type=Path, help="Input telemetry CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default="data/processed/biometrics.csv",
        help="Output biometric CSV",
    )
    parser.add_argument("--duration", type=float, default=300, help="Duration (seconds)")
    parser.add_argument("--sampling-rate", type=int, default=50, help="Telemetry sampling rate (Hz)")
    parser.add_argument("--ecg-frequency", type=int, default=500, help="ECG sampling rate (Hz)")
    args = parser.parse_args()

    n_samples = int(args.duration * args.sampling_rate)

    logger.info(f"Generating biometric data...")
    logger.info(f"  Duration: {args.duration} s")
    logger.info(f"  Telemetry sampling: {args.sampling_rate} Hz")
    logger.info(f"  ECG sampling: {args.ecg_frequency} Hz")

    timestamps, ecg, hr, hrv, stress = generate_biometric_dataset(
        telemetry_path=args.telemetry,
        n_samples=n_samples,
        sampling_rate=args.sampling_rate,
        ecg_sampling_rate=args.ecg_frequency,
    )

    save_biometric_data(args.output, timestamps, ecg, hr, hrv, stress)
    logger.info("✓ Biometric generation complete")


if __name__ == "__main__":
    main()
