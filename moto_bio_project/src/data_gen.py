"""
Synthetic Telemetry Generation Module
Physics-based motorcycle racing simulation with NeuroKit2 ECG integration
"""

import numpy as np
import pandas as pd
import neurokit2 as nk
from pathlib import Path
from typing import Tuple, Dict, List
from dataclasses import dataclass
import json
import sys

# Soporte para importación relativa y absoluta
try:
    from .config import SIM_CONFIG, PATHS
except ImportError:
    from config import SIM_CONFIG, PATHS


@dataclass
class RaceSessionData:
    """Container for race session telemetry and physiological data"""
    telemetry_df: pd.DataFrame
    ecg_signal: np.ndarray
    hrv_metrics: Dict[str, float]
    metadata: Dict[str, any]


class SyntheticTelemetry:
    """
    Generates realistic motorcycle racing telemetry using physics models
    and correlates it with physiological signals (ECG, HRV)
    """
    
    def __init__(self, config: object = SIM_CONFIG):
        """
        Initialize telemetry generator
        
        Args:
            config: SimulationConfig object with parameters
        """
        self.config = config
        self.total_samples = None
        self.time_array = None
        
    def _simulate_circuit_lap(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate a single lap around the circuit with realistic physics
        
        Returns:
            Tuple of (time, speed, lean_angle, g_force) arrays
        """
        lap_duration = self.config.LAP_DURATION_SECONDS
        sampling_rate = self.config.SAMPLING_RATE_TELEMETRY
        num_samples = int(lap_duration * sampling_rate)
        
        time = np.linspace(0, lap_duration, num_samples)
        
        # Circuit profile: 2 straights + 4 corners
        # Assumption: every 15 seconds = 1 corner section
        corner_times = np.array([15, 30, 45])  # seconds when entering corners
        
        # Speed profile: accelerate on straights, brake into corners
        speed = np.zeros_like(time, dtype=float)
        for i, t in enumerate(time):
            # Find position in lap
            phase = (t % 60) / 15  # 4 segments of 15s each
            
            if phase < 1:  # First straight
                speed[i] = 200 + 100 * phase
            elif phase < 1.5:  # First corner
                speed[i] = 300 - 200 * (phase - 1) / 0.5
            elif phase < 2:  # Second straight
                speed[i] = 100 + 150 * (phase - 1.5) / 0.5
            elif phase < 2.5:  # Second corner
                speed[i] = 250 - 150 * (phase - 2) / 0.5
            elif phase < 3:  # Third straight
                speed[i] = 100 + 180 * (phase - 2.5) / 0.5
            elif phase < 3.5:  # Third corner
                speed[i] = 280 - 200 * (phase - 3) / 0.5
            else:  # Fourth straight
                speed[i] = 80 + 200 * (phase - 3.5) / 0.5
        
        speed = np.clip(speed, self.config.MIN_SPEED_KMH, self.config.MAX_SPEED_KMH)
        
        # Lean angle: proportional to speed in corners (physics)
        # High speed in corner = high lean angle
        lean_angle = np.zeros_like(time, dtype=float)
        for i, (t, s) in enumerate(zip(time, speed)):
            phase = (t % 60) / 15
            
            # Lean only in corners (phases 1-1.5, 2.5-3, 3.5-4)
            in_corner = (0.8 < phase < 1.5) or (2.3 < phase < 3.0) or (3.3 < phase < 4.0)
            
            if in_corner:
                lean_angle[i] = (s / self.config.MAX_SPEED_KMH) * self.config.MAX_LEAN_ANGLE
            else:
                lean_angle[i] = 0.0
        
        # G-force: from lateral acceleration and braking/acceleration
        g_force = np.zeros_like(time, dtype=float)
        for i, (t, s, la) in enumerate(zip(time, speed, lean_angle)):
            # Lateral G from lean angle
            lateral_g = (la / 90.0) * self.config.MAX_G_FORCE
            
            # Longitudinal G from speed changes
            if i > 0:
                speed_diff = (speed[i] - speed[i-1]) / (1/self.config.SAMPLING_RATE_TELEMETRY)  # km/h per second
                accel_g = (speed_diff / 100.0) * 0.3  # Scale down acceleration
            else:
                accel_g = 0.0
            
            g_force[i] = np.sqrt(lateral_g**2 + accel_g**2)
            g_force[i] = np.clip(g_force[i], 0, self.config.MAX_G_FORCE)
        
        return time, speed, lean_angle, g_force
    
    def _generate_heart_rate_profile(self, g_force: np.ndarray, speed: np.ndarray) -> np.ndarray:
        """
        Generate heart rate profile correlated with physical stress (G-force + speed)
        
        Args:
            g_force: G-force array
            speed: Speed array in km/h
            
        Returns:
            Heart rate array in bpm
        """
        # Base HR response to G-force (sympathetic activation)
        stress_signal = g_force + (speed / self.config.MAX_SPEED_KMH) * 0.5
        
        # Exponential smoothing to simulate physiological lag
        tau = self.config.HR_RESPONSE_LAG * self.config.SAMPLING_RATE_TELEMETRY
        alpha = 1.0 / tau
        
        hr_response = np.zeros_like(stress_signal)
        hr_response[0] = self.config.HR_BASELINE
        
        for i in range(1, len(stress_signal)):
            # HR = baseline + stress * sensitivity
            stress_factor = stress_signal[i] / (self.config.MAX_G_FORCE + 0.5)
            target_hr = self.config.HR_BASELINE + stress_factor * self.config.HR_PER_G_FORCE
            target_hr = np.clip(target_hr, self.config.RESTING_HR, self.config.MAX_HR)
            
            # Apply exponential smoothing
            hr_response[i] = alpha * target_hr + (1 - alpha) * hr_response[i-1]
        
        # Add sinusoidal variability (natural HRV)
        hrv_noise = np.random.normal(0, 2, len(hr_response))
        hr_response = hr_response + hrv_noise
        hr_response = np.clip(hr_response, self.config.RESTING_HR, self.config.MAX_HR)
        
        return hr_response
    
    def _generate_ecg_signal(self, heart_rate: np.ndarray, duration_seconds: int) -> np.ndarray:
        """
        Generate realistic ECG signal using NeuroKit2
        
        Args:
            heart_rate: Array of heart rates (bpm)
            duration_seconds: Total duration of ECG signal
            
        Returns:
            ECG signal array at ECG_SAMPLING_RATE
        """
        # NeuroKit2: Generate ECG at specified sampling rate
        # Use mean HR for generation, then apply variability
        mean_hr = np.mean(heart_rate)
        
        ecg_signal = nk.ecg_simulate(
            duration=duration_seconds,
            sampling_rate=self.config.ECG_SAMPLING_RATE,
            heart_rate=int(mean_hr),
            method="simple"
        )
        
        return ecg_signal
    
    def _compute_hrv_metrics(self, ecg_signal: np.ndarray) -> Dict[str, float]:
        """
        Compute HRV metrics from ECG signal (RMSSD, SDNN, etc.)
        
        Args:
            ecg_signal: Raw ECG signal
            
        Returns:
            Dictionary with HRV metrics
        """
        # Process ECG to find R-peaks
        signals, info = nk.ecg_process(
            ecg_signal,
            sampling_rate=self.config.ECG_SAMPLING_RATE
        )
        
        # Calculate HRV
        hrv_metrics = nk.hrv_frequency(
            signals,
            sampling_rate=self.config.ECG_SAMPLING_RATE
        )
        
        # Extract key metrics - usar .iloc[0] para evitar FutureWarning
        def safe_float(value):
            """Convertir Series o valor a float de forma segura"""
            if hasattr(value, 'iloc'):
                return float(value.iloc[0])
            return float(value) if value is not None else 0.0
        
        metrics = {
            "rmssd": safe_float(hrv_metrics.get("HRV_RMSSD", 0.0)),
            "sdnn": safe_float(hrv_metrics.get("HRV_SDNN", 0.0)),
            "pnn50": safe_float(hrv_metrics.get("HRV_pNN50", 0.0)),
            "hf": safe_float(hrv_metrics.get("HRV_HF", 0.0)),
            "lf": safe_float(hrv_metrics.get("HRV_LF", 0.0)),
        }
        
        return metrics
    
    def _add_vibration_noise(self, ecg_signal: np.ndarray, speed: np.ndarray) -> np.ndarray:
        """
        Add realistic ECG artifacts from motorcycle vibrations at high speed
        
        Args:
            ecg_signal: Clean ECG signal
            speed: Speed array (for interpolation to ECG sampling rate)
            
        Returns:
            ECG signal with vibration noise
        """
        # Interpolate speed to ECG sampling rate
        num_ecg_samples = len(ecg_signal)
        speed_ecg = np.interp(
            np.linspace(0, len(speed)-1, num_ecg_samples),
            np.arange(len(speed)),
            speed
        )
        
        # Vibration noise proportional to speed
        vibration_magnitude = (speed_ecg / self.config.MAX_SPEED_KMH) * 0.05  # 5% max noise
        vibration_noise = np.random.normal(0, vibration_magnitude)
        
        ecg_with_noise = ecg_signal + vibration_noise
        
        return ecg_with_noise
    
    def generate_race_session(self, n_laps: int = None, output_dir: Path = None) -> RaceSessionData:
        """
        Generate complete race session with multiple laps
        
        Args:
            n_laps: Number of laps to simulate (default from config)
            output_dir: Directory to save outputs
            
        Returns:
            RaceSessionData object with telemetry and physiological data
        """
        if n_laps is None:
            n_laps = self.config.NUM_LAPS
        if output_dir is None:
            output_dir = PATHS.DATA_DIR
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🏍️ Generating {n_laps} laps of racing data...")
        
        all_telemetry = []
        all_ecg = []
        all_hr = []
        
        for lap in range(n_laps):
            if (lap + 1) % 10 == 0:
                print(f"   Lap {lap + 1}/{n_laps}")
            
            # Generate single lap
            time, speed, lean_angle, g_force = self._simulate_circuit_lap()
            
            # Fatigue effect: HR increases over laps
            fatigue_factor = 1.0 + (lap / n_laps) * 0.2
            
            # Generate HR profile
            heart_rate = self._generate_heart_rate_profile(g_force, speed) * fatigue_factor
            heart_rate = np.clip(heart_rate, self.config.RESTING_HR, self.config.MAX_HR)
            
            # Add lap offset to time
            time_offset = lap * self.config.LAP_DURATION_SECONDS
            time = time + time_offset
            
            # Create telemetry dataframe for this lap
            lap_data = pd.DataFrame({
                'time_s': time,
                'speed_kmh': speed,
                'lean_angle_deg': lean_angle,
                'g_force': g_force,
                'heart_rate_bpm': heart_rate,
                'lap': lap,
            })
            
            all_telemetry.append(lap_data)
            all_hr.extend(heart_rate)
        
        # Combine all laps
        telemetry_df = pd.concat(all_telemetry, ignore_index=True)
        
        # Generate ECG signal for entire session
        total_duration = n_laps * self.config.LAP_DURATION_SECONDS
        ecg_signal = self._generate_ecg_signal(np.array(all_hr), total_duration)
        
        # Add vibration noise
        speed_array = telemetry_df['speed_kmh'].values
        ecg_signal = self._add_vibration_noise(ecg_signal, speed_array)
        
        # Compute HRV metrics
        hrv_metrics = self._compute_hrv_metrics(ecg_signal)
        
        # Metadata
        metadata = {
            "num_laps": n_laps,
            "total_duration_seconds": float(total_duration),
            "circuit_length_km": float(self.config.CIRCUIT_LENGTH_KM),
            "sampling_rate_telemetry_hz": int(self.config.SAMPLING_RATE_TELEMETRY),
            "sampling_rate_ecg_hz": int(self.config.ECG_SAMPLING_RATE),
            "mean_speed_kmh": float(telemetry_df['speed_kmh'].mean()),
            "max_speed_kmh": float(telemetry_df['speed_kmh'].max()),
            "mean_hr_bpm": float(telemetry_df['heart_rate_bpm'].mean()),
            "max_hr_bpm": float(telemetry_df['heart_rate_bpm'].max()),
        }
        
        # Save outputs
        telemetry_csv = output_dir / "telemetry.csv"
        ecg_file = output_dir / "ecg_signal.npy"
        hrv_file = output_dir / "hrv_metrics.json"
        metadata_file = output_dir / "metadata.json"
        
        telemetry_df.to_csv(telemetry_csv, index=False)
        np.save(ecg_file, ecg_signal)
        with open(hrv_file, 'w') as f:
            json.dump(hrv_metrics, f, indent=2)
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Data saved to {output_dir}")
        
        return RaceSessionData(
            telemetry_df=telemetry_df,
            ecg_signal=ecg_signal,
            hrv_metrics=hrv_metrics,
            metadata=metadata
        )


def main():
    """Generate sample race session"""
    gen = SyntheticTelemetry()
    session = gen.generate_race_session(n_laps=10)
    print(f"\n📊 Session Summary:")
    print(f"   Total laps: {session.metadata['num_laps']}")
    print(f"   Duration: {session.metadata['total_duration_seconds']:.1f}s")
    print(f"   Mean speed: {session.metadata['mean_speed_kmh']:.1f} km/h")
    print(f"   Mean HR: {session.metadata['mean_hr_bpm']:.1f} bpm")
    print(f"   HRV (RMSSD): {session.hrv_metrics['rmssd']:.1f} ms")


if __name__ == "__main__":
    main()
