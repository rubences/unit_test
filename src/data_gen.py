"""
Phase 1: Synthetic Data Generation for Bio-Adaptive Haptic Coaching

This module simulates realistic telemetry data from a motorcycle racing session,
including physics-based speed/G-force dynamics and biometric correlations.

Key Innovation:
- Realistic ECG simulation via NeuroKit2
- Physiological correlation: HR increases with G-Force and stress
- Noise injection during high-speed segments (vibration artifacts)
"""

import numpy as np
import pandas as pd
import neurokit2 as nk
import os
from pathlib import Path


def simulate_circuit_lap(lap_duration=60, sampling_rate=100):
    """
    Simulate a single motorcycle racing lap with realistic telemetry.
    
    Physics Model:
    - Circuit: 1.2 km with 4 corners (high G-force) and 2 straights (high speed)
    - Speed: 50-300 km/h based on corner/straight sections
    - G-Force: 0.2-2.0 G based on lean angle and speed
    - Lean Angle: 0-63 degrees (realistic MotoGP range)
    
    Args:
        lap_duration (int): Lap time in seconds (realistic for 1.2 km circuit)
        sampling_rate (int): Hz, data points per second
    
    Returns:
        dict: Telemetry data with keys 'time', 'speed', 'g_force', 'lean_angle'
    """
    n_points = lap_duration * sampling_rate
    time = np.linspace(0, lap_duration, n_points)
    
    # Define circuit sections (4 laps of acceleration/corner pattern)
    # Straight: 0-15s, Corner 1: 15-20s, Straight: 20-35s, Corner 2: 35-40s, repeat
    section_pattern = (time % 40) / 40
    
    # Speed dynamics (km/h)
    # Straights: 250-300 km/h, Corners: 50-150 km/h
    is_straight = ((section_pattern < 0.375) | (section_pattern > 0.875))
    speed = np.where(
        is_straight,
        250 + 50 * np.sin(section_pattern * 2 * np.pi),  # Smooth accel/decel in straights
        100 + 50 * np.cos(section_pattern * 2 * np.pi)   # Lower in corners
    )
    
    # G-Force (based on lean angle and speed)
    # Corner G-force: proportional to (Speed^2 / Radius) and lean angle
    # Straight G-force: mainly braking/acceleration
    lean_angle = np.where(
        is_straight,
        5 + 10 * np.abs(np.sin(section_pattern * 2 * np.pi)),  # Small lean in straights
        20 + 30 * np.abs(np.sin(section_pattern * 2 * np.pi))  # Large lean in corners
    )
    
    # G-Force = lateral accel (from lean) + longitudinal accel
    g_force = (
        1.5 * np.sin(lean_angle * np.pi / 180) +  # Lateral G from lean angle
        0.5 * np.abs(np.sin(section_pattern * np.pi))  # Longitudinal G
    )
    g_force = np.clip(g_force, 0.2, 2.5)  # Realistic bounds
    
    return {
        'time': time,
        'speed': speed,
        'g_force': g_force,
        'lean_angle': lean_angle
    }


def generate_heart_rate_profile(g_force, speed, sampling_rate=100, baseline_hr=70):
    """
    Generate physiologically realistic heart rate based on physical stress.
    
    Biophysical Model:
    - Baseline HR (resting): 70 bpm
    - HR increases linearly with G-Force (sympathetic activation)
    - HR increases with speed (cognitive stress)
    - Variance increases at high stress (autonomic instability)
    - Refractory period: HR doesn't drop instantly (parasympathetic lag)
    
    Args:
        g_force (array): G-force profile
        speed (array): Speed profile (km/h)
        sampling_rate (int): Hz
        baseline_hr (float): Resting heart rate (bpm)
    
    Returns:
        array: Heart rate (bpm) at each timepoint
    """
    n_points = len(g_force)
    
    # Stress metric: normalized combination of G-force and speed
    # G-force range: 0.2-2.5 G, Speed range: 50-300 km/h
    normalized_g = (g_force - 0.2) / (2.5 - 0.2)
    normalized_speed = (speed - 50) / (300 - 50)
    stress_metric = 0.6 * normalized_g + 0.4 * normalized_speed  # Mostly physical, some cognitive
    
    # HR response: baseline + stress response (HR max ~180 for trained riders)
    target_hr = baseline_hr + stress_metric * (180 - baseline_hr)
    
    # First-order exponential lag (parasympathetic response time constant ~5 seconds)
    tau_seconds = 5  # Time constant for HR to track target
    dt = 1 / sampling_rate
    alpha = dt / (tau_seconds + dt)
    
    actual_hr = np.zeros(n_points)
    actual_hr[0] = baseline_hr
    for i in range(1, n_points):
        actual_hr[i] = actual_hr[i-1] + alpha * (target_hr[i] - actual_hr[i-1])
    
    return actual_hr


def generate_ecg_signal(heart_rate, duration=60, sampling_rate=500):
    """
    Generate realistic ECG signal using NeuroKit2.
    
    Technical Notes:
    - NeuroKit2 uses sinusoidal ECG model with morphology variations
    - Heart rate varies according to input array
    - Sampling rate higher than telemetry (500 Hz ECG vs 100 Hz telemetry)
    
    Args:
        heart_rate (array): HR profile (bpm) at telemetry rate (100 Hz)
        duration (int): Signal duration (seconds)
        sampling_rate (int): ECG sampling rate (Hz)
    
    Returns:
        array: ECG signal (mV) at 500 Hz
    """
    n_ecg_points = duration * sampling_rate
    
    # Interpolate HR to ECG sampling rate
    telemetry_time = np.linspace(0, duration, len(heart_rate))
    ecg_time = np.linspace(0, duration, n_ecg_points)
    hr_at_ecg_rate = np.interp(ecg_time, telemetry_time, heart_rate)
    
    # Generate ECG using NeuroKit2
    ecg_signal = nk.ecg_simulate(
        duration=duration,
        sampling_rate=sampling_rate,
        heart_rate=hr_at_ecg_rate,
        method='simple'
    )
    
    return ecg_signal


def compute_hrv_metrics(ecg_signal, sampling_rate=500):
    """
    Compute Heart Rate Variability metrics from ECG signal.
    
    Key Metric: RMSSD (Root Mean Square of Successive Differences)
    - RMSSD reflects parasympathetic (vagal) tone
    - High RMSSD (>30 ms): Low stress, high parasympathetic
    - Low RMSSD (<20 ms): High stress, sympathetic dominance
    - RMSSD range in cyclists: 10-100 ms
    
    Args:
        ecg_signal (array): ECG signal in mV
        sampling_rate (int): ECG sampling rate (Hz)
    
    Returns:
        dict: HRV metrics including RMSSD and HF/LF ratio
    """
    # Detect R-peaks
    _, peaks = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate)
    
    if len(peaks['ECG_R_Peaks']) < 2:
        return {'rmssd': 50.0, 'hf_lf_ratio': 1.0, 'n_peaks': 0}
    
    # Compute inter-beat intervals (RR intervals in milliseconds)
    peak_times = peaks['ECG_R_Peaks'] / sampling_rate * 1000  # Convert to ms
    rr_intervals = np.diff(peak_times)
    
    # RMSSD = sqrt(mean(RR_diff^2))
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
    
    # Normalize: typical range 10-100 ms, we map to 0-1 stress scale
    # Low RMSSD (20 ms) = high stress, High RMSSD (60 ms) = low stress
    hrv_index = (rmssd - 20) / (60 - 20)  # Normalized to 0-1
    hrv_index = np.clip(hrv_index, 0, 1)
    
    # Simple HF/LF ratio (without formal FFT, as approximation)
    # In real ECG: HF (0.15-0.4 Hz) = parasympathetic, LF (0.04-0.15 Hz) = sympathetic
    hf_lf_ratio = 1.0 + hrv_index * 2.0  # Range 1.0-3.0
    
    return {
        'rmssd': float(rmssd),
        'hf_lf_ratio': float(hf_lf_ratio),
        'n_peaks': len(peaks['ECG_R_Peaks'])
    }


def add_vibration_noise(ecg_signal, speed, sampling_rate=500):
    """
    Add realistic vibration noise to ECG signal (motorcycle vibration artifact).
    
    Vibration Model:
    - High speed (>200 km/h): More vibration noise (higher frequency muscle artifact)
    - Noise amplitude scales with speed squared (aerodynamic buffeting)
    - Frequency content: 50-200 Hz (typical motorcycle vibration)
    
    Args:
        ecg_signal (array): Clean ECG signal
        speed (array): Speed profile (downsampled to ECG rate)
        sampling_rate (int): ECG sampling rate (Hz)
    
    Returns:
        array: Noisy ECG signal
    """
    n_points = len(ecg_signal)
    duration = n_points / sampling_rate
    
    # Interpolate speed to ECG rate
    telemetry_duration = len(speed) / 100  # speed is at 100 Hz
    speed_at_ecg = np.interp(
        np.linspace(0, duration, n_points),
        np.linspace(0, telemetry_duration, len(speed)),
        speed
    )
    
    # Vibration amplitude scales with (speed / max_speed)^2
    normalized_speed = (speed_at_ecg - 50) / (300 - 50)
    vibration_amplitude = np.clip(normalized_speed ** 2, 0, 0.3)  # Max 0.3 mV noise
    
    # Simulate muscle artifact (50-200 Hz noise)
    noise_freq = 100  # Hz (typical muscle artifact frequency)
    time = np.linspace(0, duration, n_points)
    muscle_artifact = np.sin(2 * np.pi * noise_freq * time)
    
    # Modulate noise amplitude by speed
    noisy_ecg = ecg_signal + vibration_amplitude * muscle_artifact * 0.1
    
    return noisy_ecg


def generate_race_session(n_laps=10, sampling_rate_telemetry=100, 
                         lap_duration=60, output_dir='data/raw'):
    """
    Main function: Generate a complete race session with synchronized telemetry and ECG.
    
    Args:
        n_laps (int): Number of laps to simulate
        sampling_rate_telemetry (int): Hz, telemetry sampling rate
        lap_duration (int): Seconds per lap
        output_dir (str): Directory to save .csv and .npz files
    
    Returns:
        tuple: (df_telemetry, ecg_full, dict_metadata)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Pre-allocate arrays for all laps
    total_duration = n_laps * lap_duration
    n_telem_points = total_duration * sampling_rate_telemetry
    
    # Telemetry arrays
    speed_full = np.zeros(n_telem_points)
    g_force_full = np.zeros(n_telem_points)
    lean_angle_full = np.zeros(n_telem_points)
    hr_full = np.zeros(n_telem_points)
    
    # ECG array (at 500 Hz)
    ecg_sampling_rate = 500
    n_ecg_points = total_duration * ecg_sampling_rate
    ecg_full = np.zeros(n_ecg_points)
    
    print(f"Generating {n_laps} laps of telemetry data...")
    
    # Generate lap by lap
    for lap in range(n_laps):
        print(f"  Lap {lap + 1}/{n_laps}...", end=' ')
        
        # Simulate lap telemetry
        lap_data = simulate_circuit_lap(lap_duration=lap_duration, 
                                        sampling_rate=sampling_rate_telemetry)
        
        # Extract indices for this lap
        start_idx = lap * lap_duration * sampling_rate_telemetry
        end_idx = (lap + 1) * lap_duration * sampling_rate_telemetry
        
        # Store telemetry
        speed_full[start_idx:end_idx] = lap_data['speed']
        g_force_full[start_idx:end_idx] = lap_data['g_force']
        lean_angle_full[start_idx:end_idx] = lap_data['lean_angle']
        
        # Generate HR based on G-force and speed
        hr_lap = generate_heart_rate_profile(
            lap_data['g_force'],
            lap_data['speed'],
            sampling_rate=sampling_rate_telemetry
        )
        hr_full[start_idx:end_idx] = hr_lap
        
        # Generate ECG for this lap
        ecg_lap = generate_ecg_signal(
            hr_lap,
            duration=lap_duration,
            sampling_rate=ecg_sampling_rate
        )
        
        # Add vibration noise to ECG
        ecg_lap_noisy = add_vibration_noise(
            ecg_lap,
            lap_data['speed'],
            sampling_rate=ecg_sampling_rate
        )
        
        # Store ECG
        start_ecg_idx = lap * lap_duration * ecg_sampling_rate
        end_ecg_idx = (lap + 1) * lap_duration * ecg_sampling_rate
        ecg_full[start_ecg_idx:end_ecg_idx] = ecg_lap_noisy
        
        print("✓")
    
    # Create time axis
    time_full = np.linspace(0, total_duration, n_telem_points)
    
    # Create pandas DataFrame for telemetry
    df_telemetry = pd.DataFrame({
        'timestamp': time_full,
        'speed_kmh': speed_full,
        'g_force': g_force_full,
        'lean_angle_deg': lean_angle_full,
        'heart_rate_bpm': hr_full
    })
    
    # Compute HRV metrics for each lap
    hrv_metrics_list = []
    for lap in range(n_laps):
        start_ecg = lap * lap_duration * ecg_sampling_rate
        end_ecg = (lap + 1) * lap_duration * ecg_sampling_rate
        ecg_lap = ecg_full[start_ecg:end_ecg]
        
        hrv = compute_hrv_metrics(ecg_lap, sampling_rate=ecg_sampling_rate)
        hrv_metrics_list.append(hrv)
    
    # Metadata
    metadata = {
        'n_laps': n_laps,
        'lap_duration_sec': lap_duration,
        'total_duration_sec': total_duration,
        'sampling_rate_telemetry_hz': sampling_rate_telemetry,
        'sampling_rate_ecg_hz': ecg_sampling_rate,
        'circuit_length_km': 1.2,
        'avg_speed_kmh': float(np.mean(speed_full)),
        'avg_g_force': float(np.mean(g_force_full)),
        'avg_hr_bpm': float(np.mean(hr_full)),
        'hrv_metrics_per_lap': hrv_metrics_list
    }
    
    # Save telemetry to CSV
    csv_path = os.path.join(output_dir, 'race_telemetry.csv')
    df_telemetry.to_csv(csv_path, index=False)
    print(f"\n✓ Telemetry saved to {csv_path}")
    
    # Save ECG signal to NPZ (binary, preserves precision)
    npz_path = os.path.join(output_dir, 'race_ecg.npz')
    np.savez(
        npz_path,
        ecg_signal=ecg_full,
        sampling_rate=ecg_sampling_rate,
        duration=total_duration,
        n_laps=n_laps
    )
    print(f"✓ ECG signal saved to {npz_path}")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, 'metadata.txt')
    with open(metadata_path, 'w') as f:
        f.write("=== Race Session Metadata ===\n\n")
        f.write(f"Total Duration: {metadata['total_duration_sec']} seconds ({n_laps} laps)\n")
        f.write(f"Telemetry Sampling Rate: {metadata['sampling_rate_telemetry_hz']} Hz\n")
        f.write(f"ECG Sampling Rate: {metadata['sampling_rate_ecg_hz']} Hz\n")
        f.write(f"\nAverage Metrics:\n")
        f.write(f"  Speed: {metadata['avg_speed_kmh']:.1f} km/h\n")
        f.write(f"  G-Force: {metadata['avg_g_force']:.2f} G\n")
        f.write(f"  Heart Rate: {metadata['avg_hr_bpm']:.1f} bpm\n")
        f.write(f"\nHRV Metrics (per lap):\n")
        for lap, hrv in enumerate(metadata['hrv_metrics_per_lap']):
            f.write(f"  Lap {lap+1}: RMSSD={hrv['rmssd']:.2f} ms, HF/LF={hrv['hf_lf_ratio']:.2f}\n")
    
    print(f"✓ Metadata saved to {metadata_path}")
    
    print(f"\n=== Data Generation Summary ===")
    print(f"Laps: {n_laps}")
    print(f"Telemetry points: {len(df_telemetry):,}")
    print(f"ECG points: {len(ecg_full):,}")
    print(f"Avg Speed: {metadata['avg_speed_kmh']:.1f} km/h")
    print(f"Avg HR: {metadata['avg_hr_bpm']:.1f} bpm")
    print()
    
    return df_telemetry, ecg_full, metadata


if __name__ == '__main__':
    # Generate a 10-lap race session
    df, ecg, meta = generate_race_session(n_laps=10, output_dir='data/raw')
    print("Data generation complete!")
