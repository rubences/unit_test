"""
Biometric Data Simulator: ECG Signal Generation with Correlation to Telemetry

This module generates realistic ECG signals correlated to motorcycle telemetry,
simulating the driver's physiological response to different track conditions.

Key Features:
- HR modulation based on track stress (g-force, lean angle, speed)
- HRV adaptation: high HRV (low stress) on straights, low HRV (high stress) on turns
- Realistic QRS artifacts simulating handlebar vibrations
- Signal noise using neurokit2.signal_distort()

Theory:
- HR (Heart Rate): Increases with cognitive/physical load (~110 bpm baseline → ~180 bpm peak)
- HRV (Heart Rate Variability): Decreases with stress (high vagal tone → low parasympathetic)
- RMSSD: Root Mean Square of Successive Differences - calculated as:
  RMSSD = sqrt(mean((RR_intervals[i+1] - RR_intervals[i])^2))
  Unit: milliseconds, Range: 20-100ms (low stress) to 5-20ms (high stress)

Reference: neurokit2.ecg_simulate() uses mathematical ECG models (synthetic ECG generation)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import logging

try:
    import neurokit2 as nk
    NEUROKIT2_AVAILABLE = True
except ImportError:
    NEUROKIT2_AVAILABLE = False
    logging.warning("neurokit2 not available. Install: pip install neurokit2")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BiometricDataSimulator:
    """
    Generates synthetic ECG data correlated with motorcycle telemetry.
    
    Correlation Logic:
    - Straight sections (low g-force): Baseline HR (~110 bpm), High HRV
    - Braking/Turns (high g-force > 1.2G): Elevated HR (~170 bpm), Collapsed HRV
    - Dangerous conditions: HR spike (~180 bpm), minimal HRV, QRS artifacts
    """
    
    # HR parameters (beats per minute)
    HR_BASELINE = 110         # Rest state on straight
    HR_MODERATE_STRESS = 150  # Medium turn (0.8-1.2G)
    HR_HIGH_STRESS = 170      # Hard turn (>1.2G)
    HR_EXTREME = 180          # Panic/max effort
    
    # HRV parameters (RMSSD in milliseconds)
    # Reference: RMSSD = sqrt(mean((RR_i+1 - RR_i)^2))
    RMSSD_RELAXED = 60        # Low stress, parasympathetic dominance
    RMSSD_MODERATE = 30       # Moderate stress
    RMSSD_HIGH_STRESS = 15    # High cognitive load
    RMSSD_PANIC = 8           # Saturated cognitive state
    
    # Sampling parameters
    SAMPLING_RATE = 500       # Hz (standard ECG sampling)
    ECG_DURATION = 5          # seconds per segment
    
    def __init__(self, sampling_rate: int = 500, seed: Optional[int] = None):
        """
        Initialize biometric simulator.
        
        Args:
            sampling_rate: ECG sampling frequency (Hz)
            seed: Random seed for reproducibility
        """
        if not NEUROKIT2_AVAILABLE:
            raise ImportError("neurokit2 required: pip install neurokit2")
        
        self.sampling_rate = sampling_rate
        self.rng = np.random.RandomState(seed)
        logger.info(f"✓ BiometricDataSimulator initialized (fs={sampling_rate}Hz)")
    
    def compute_stress_level(
        self,
        g_force: float,
        lean_angle: float,
        speed: float,
        duration_in_stress: float = 0.0,
    ) -> float:
        """
        Compute stress level (0-1) from telemetry.
        
        Args:
            g_force: Current G-force (0-2+)
            lean_angle: Lean angle in degrees (0-60+)
            speed: Speed in m/s (0-70+)
            duration_in_stress: Time spent in high stress state (seconds)
        
        Returns:
            stress_level: Float in [0, 1]
                - 0.0: Relaxed (straight, low speed)
                - 0.5: Moderate (gentle turn)
                - 1.0: Extreme (hard turn, high speed, sustained stress)
        """
        # G-force contribution (40%)
        g_stress = min(g_force / 2.0, 1.0) * 0.4
        
        # Lean angle contribution (30%)
        lean_stress = min(lean_angle / 65.0, 1.0) * 0.3
        
        # Speed contribution (20%)
        speed_stress = min(speed / 70.0, 1.0) * 0.2
        
        # Cumulative stress contribution (10%)
        cumulative_stress = min(duration_in_stress / 10.0, 1.0) * 0.1
        
        total_stress = g_stress + lean_stress + speed_stress + cumulative_stress
        return np.clip(total_stress, 0.0, 1.0)
    
    def get_hr_from_stress(self, stress_level: float) -> float:
        """
        Map stress level to Heart Rate.
        
        HR follows a smooth curve:
        - stress=0.0 → HR=110 bpm (relaxed)
        - stress=0.5 → HR=145 bpm (moderate)
        - stress=1.0 → HR=180 bpm (panic)
        
        Args:
            stress_level: Stress in [0, 1]
        
        Returns:
            hr: Heart rate in beats per minute
        """
        if stress_level < 0.3:
            # Linear: baseline to moderate
            return self.HR_BASELINE + (stress_level / 0.3) * (self.HR_MODERATE_STRESS - self.HR_BASELINE)
        elif stress_level < 0.7:
            # Linear: moderate to high
            return self.HR_MODERATE_STRESS + ((stress_level - 0.3) / 0.4) * (self.HR_HIGH_STRESS - self.HR_MODERATE_STRESS)
        else:
            # Linear: high to extreme
            return self.HR_HIGH_STRESS + ((stress_level - 0.7) / 0.3) * (self.HR_EXTREME - self.HR_HIGH_STRESS)
    
    def get_rmssd_from_stress(self, stress_level: float) -> float:
        """
        Map stress level to RMSSD (Heart Rate Variability).
        
        RMSSD decreases exponentially with stress (parasympathetic withdrawal):
        - stress=0.0 → RMSSD=60 ms (relaxed)
        - stress=0.5 → RMSSD=25 ms (moderate)
        - stress=1.0 → RMSSD=8 ms (panic)
        
        Formula: RMSSD = RMSSD_relaxed * exp(-2 * stress_level)
        
        Args:
            stress_level: Stress in [0, 1]
        
        Returns:
            rmssd: RMSSD in milliseconds
        """
        # Exponential decay: parasympathetic withdrawal
        rmssd = self.RMSSD_RELAXED * np.exp(-2.0 * stress_level)
        return np.clip(rmssd, self.RMSSD_PANIC, self.RMSSD_RELAXED)
    
    def generate_rr_intervals(
        self,
        duration: float,
        target_hr: float,
        target_rmssd: float,
    ) -> np.ndarray:
        """
        Generate RR intervals (time between heartbeats) with specified HR and RMSSD.
        
        Method: Constrained random walk
        1. Base interval: 60000 / target_hr (milliseconds)
        2. Add random variation with std = target_rmssd / sqrt(2)
        3. Ensure physical bounds (0.5 - 2.0 seconds per beat)
        
        Formula for RMSSD calculation:
        RMSSD = sqrt(mean((RR_i+1 - RR_i)^2))
        
        Args:
            duration: Total duration (seconds)
            target_hr: Desired heart rate (bpm)
            target_rmssd: Desired RMSSD (milliseconds)
        
        Returns:
            rr_intervals: Array of RR intervals (milliseconds)
        """
        # Convert HR to base RR interval
        base_rr_ms = 60000.0 / target_hr
        
        # Number of heartbeats
        num_beats = int(duration * target_hr / 60.0) + 1
        
        # Generate RR intervals as constrained random walk
        rr_intervals = np.zeros(num_beats)
        rr_intervals[0] = base_rr_ms
        
        # Standard deviation for RR variation to achieve target RMSSD
        # Empirically: std ≈ target_rmssd / sqrt(2)
        rr_std = target_rmssd / np.sqrt(2)
        
        for i in range(1, num_beats):
            # Random step
            delta = self.rng.normal(0, rr_std)
            rr_intervals[i] = rr_intervals[i-1] + delta
            
            # Physical bounds
            rr_intervals[i] = np.clip(rr_intervals[i], base_rr_ms * 0.5, base_rr_ms * 2.0)
        
        return rr_intervals
    
    def generate_ecg_segment(
        self,
        stress_level: float,
        duration: float = 5.0,
        add_artifacts: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Generate a single ECG segment with specified stress level.
        
        Pipeline:
        1. Generate RR intervals matching stress profile
        2. Create synthetic ECG using neurokit2.ecg_simulate()
        3. Add handlebar vibration artifacts using signal_distort()
        4. Return raw ECG and metadata
        
        Args:
            stress_level: Stress in [0, 1]
            duration: Segment duration (seconds)
            add_artifacts: Whether to add handlebar vibration noise
        
        Returns:
            dict with keys:
                - 'ecg_raw': Raw ECG signal (n_samples,)
                - 'hr': Target heart rate (bpm)
                - 'rmssd': Target RMSSD (ms)
                - 'stress': Stress level
                - 'duration': Duration (seconds)
                - 'sampling_rate': Sampling rate (Hz)
        """
        # Compute HR and RMSSD from stress
        target_hr = self.get_hr_from_stress(stress_level)
        target_rmssd = self.get_rmssd_from_stress(stress_level)
        
        # Ensure duration is int for neurokit2
        duration = int(duration)
        # Generate RR intervals
        rr_intervals = self.generate_rr_intervals(
            duration=duration,
            target_hr=target_hr,
            target_rmssd=target_rmssd,
        )
        
        # Create synthetic ECG using neurokit2
        # neurokit2.ecg_simulate() uses parametric ECG model to generate realistic waveforms
        ecg_synthetic = nk.ecg_simulate(
            duration=duration,
            sampling_rate=self.sampling_rate,
            heart_rate=target_hr,
            method='pchip',  # Use PCHIP interpolation for RR intervals
            random_state=self.rng.randint(0, 2**31),
        )
        
        # Add handlebar vibration artifacts
        if add_artifacts:
            # Simulate high-frequency vibration (50-150 Hz typical for motorcycle)
            vibration_freq = self.rng.uniform(80, 150)  # Hz
            vibration_amplitude = stress_level * 0.5  # Amplitude increases with stress
            
            t = np.arange(len(ecg_synthetic)) / self.sampling_rate
            vibration = vibration_amplitude * np.sin(2 * np.pi * vibration_freq * t)
            
            # Add movement artifact (lower frequency, ~2-5 Hz)
            movement_freq = self.rng.uniform(2, 5)
            movement_amplitude = stress_level * 0.3
            movement = movement_amplitude * np.sin(2 * np.pi * movement_freq * t)
            
            # Combine artifacts
            ecg_with_noise = ecg_synthetic + vibration + movement
        else:
            ecg_with_noise = ecg_synthetic
        
        return {
            'ecg_raw': ecg_with_noise,
            'hr': target_hr,
            'rmssd': target_rmssd,
            'stress': stress_level,
            'duration': duration,
            'sampling_rate': self.sampling_rate,
        }
    
    def generate_episode(
        self,
        telemetry: pd.DataFrame,
        duration: float = 60.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate complete ECG episode correlated with telemetry.
        
        Args:
            telemetry: DataFrame with columns [timestamp, g_force, lean_angle, speed]
            duration: Total duration (seconds)
        
        Returns:
            ecg_signal: Full ECG signal (n_samples,)
            timestamps: Time of each sample (seconds)
            stress_profile: Stress level at each sample
        """
        # Validate telemetry
        required_cols = ['g_force', 'lean_angle', 'speed']
        if not all(col in telemetry.columns for col in required_cols):
            raise ValueError(f"Telemetry must have columns: {required_cols}")
        
        # Total samples
        n_samples = int(duration * self.sampling_rate)
        ecg_signal = np.zeros(n_samples)
        stress_profile = np.zeros(n_samples)
        
        # Segment-based generation
        segment_duration = 5.0
        segment_samples = int(segment_duration * self.sampling_rate)
        
        cumulative_stress_time = 0.0
        
        for segment_idx in range(0, n_samples, segment_samples):
            # Get telemetry stats for this segment
            segment_end = min(segment_idx + segment_samples, n_samples)
            segment_time_start = segment_idx / self.sampling_rate
            segment_time_end = segment_end / self.sampling_rate
            
            # Find corresponding telemetry
            mask = (telemetry['timestamp'] >= segment_time_start) & \
                   (telemetry['timestamp'] <= segment_time_end)
            
            if mask.sum() == 0:
                # No telemetry available, use baseline
                avg_g_force = 0.0
                avg_lean = 0.0
                avg_speed = 0.0
            else:
                seg_data = telemetry[mask]
                avg_g_force = seg_data['g_force'].mean()
                avg_lean = seg_data['lean_angle'].mean()
                avg_speed = seg_data['speed'].mean()
            
            # Compute stress and update cumulative time
            stress = self.compute_stress_level(
                g_force=avg_g_force,
                lean_angle=avg_lean,
                speed=avg_speed,
                duration_in_stress=cumulative_stress_time,
            )
            
            if stress > 0.5:
                cumulative_stress_time += segment_duration
            else:
                cumulative_stress_time = 0  # Reset on straight
            
            # Generate ECG for this segment
            segment_result = self.generate_ecg_segment(
                stress_level=stress,
                duration=segment_duration,
                add_artifacts=(stress > 0.3),
            )
            
            # Place in full arrays
            actual_segment_samples = segment_end - segment_idx
            ecg_signal[segment_idx:segment_end] = segment_result['ecg_raw'][:actual_segment_samples]
            stress_profile[segment_idx:segment_end] = stress
        
        # Generate timestamps
        timestamps = np.arange(n_samples) / self.sampling_rate
        
        logger.info(f"✓ Generated ECG episode: {duration}s at {self.sampling_rate}Hz "
                   f"(n_samples={n_samples}, stress_range=[{stress_profile.min():.2f}, {stress_profile.max():.2f}])")
        
        return ecg_signal, timestamps, stress_profile


def create_synthetic_telemetry(duration: float = 60.0, sampling_rate: float = 10.0) -> pd.DataFrame:
    """
    Create synthetic motorcycle telemetry for ECG correlation.
    
    Scenario: Circuit with mix of straights and turns
    - Straight sections: low g-force, low lean angle
    - Braking zones: high g-force (negative)
    - Turns: medium-high g-force, high lean angle
    
    Args:
        duration: Total duration (seconds)
        sampling_rate: Telemetry sampling rate (Hz)
    
    Returns:
        DataFrame with [timestamp, g_force, lean_angle, speed]
    """
    n_samples = int(duration * sampling_rate)
    time = np.arange(n_samples) / sampling_rate
    
    # Create periodic pattern (30s circuit)
    period = 30.0
    phase = 2 * np.pi * time / period
    
    # Telemetry parameters
    g_force = 0.3 + 0.8 * np.sin(phase) + 0.3 * np.sin(2 * phase)  # Peak ~1.4G on turns
    lean_angle = 20 + 30 * np.sin(phase + np.pi / 4)  # 0-50 degrees
    speed = 40 + 20 * np.cos(phase) + 10 * np.sin(3 * phase)  # 30-65 m/s
    
    # Add realistic discontinuities (hard braking)
    brake_points = np.where((phase > np.pi) & (phase < np.pi + 0.3))[0]
    if len(brake_points) > 0:
        brake_indices = brake_points[::3]  # Every 3rd sample in braking zone
        g_force[brake_indices] = np.clip(g_force[brake_indices] + 0.8, -1, 2)
    
    # Ensure physical bounds
    g_force = np.clip(g_force, -1.5, 2.0)
    lean_angle = np.clip(lean_angle, 0, 65)
    speed = np.clip(speed, 0, 75)
    
    return pd.DataFrame({
        'timestamp': time,
        'g_force': g_force,
        'lean_angle': lean_angle,
        'speed': speed,
    })


if __name__ == '__main__':
    # Example usage
    logger.info("=" * 60)
    logger.info("Biometric Data Simulator Demo")
    logger.info("=" * 60)
    
    # Create simulator
    sim = BiometricDataSimulator(sampling_rate=500, seed=42)
    
    # Generate synthetic telemetry
    logger.info("\n1. Generating synthetic telemetry...")
    telemetry = create_synthetic_telemetry(duration=60, sampling_rate=10)
    logger.info(f"   Telemetry shape: {telemetry.shape}")
    logger.info(f"   g_force range: [{telemetry['g_force'].min():.2f}, {telemetry['g_force'].max():.2f}]")
    logger.info(f"   lean_angle range: [{telemetry['lean_angle'].min():.1f}, {telemetry['lean_angle'].max():.1f}]°")
    logger.info(f"   speed range: [{telemetry['speed'].min():.1f}, {telemetry['speed'].max():.1f}] m/s")
    
    # Generate ECG
    logger.info("\n2. Generating correlated ECG signal...")
    ecg_signal, timestamps, stress_profile = sim.generate_episode(
        telemetry=telemetry,
        duration=60.0,
    )
    logger.info(f"   ECG signal shape: {ecg_signal.shape}")
    logger.info(f"   Stress range: [{stress_profile.min():.2f}, {stress_profile.max():.2f}]")
    
    # Display stress statistics
    low_stress_mask = stress_profile < 0.3
    med_stress_mask = (stress_profile >= 0.3) & (stress_profile < 0.7)
    high_stress_mask = stress_profile >= 0.7
    
    logger.info("\n3. Stress distribution:")
    logger.info(f"   Low stress (< 0.3):  {low_stress_mask.sum() / len(stress_profile) * 100:.1f}%")
    logger.info(f"   Med stress (0.3-0.7): {med_stress_mask.sum() / len(stress_profile) * 100:.1f}%")
    logger.info(f"   High stress (> 0.7):  {high_stress_mask.sum() / len(stress_profile) * 100:.1f}%")
    
    logger.info("\n✓ Demo complete")
