"""
Biometric Signal Processing: Real-time ECG analysis for stress detection

This module implements signal processing pipeline using neurokit2:
1. ECG Cleaning: Remove baseline wander and noise
2. QRS Detection: Identify R-peaks for RR interval calculation
3. Feature Extraction: Compute HR, RMSSD, and stress indices

Key References:
- RMSSD = sqrt(mean((RR_i+1 - RR_i)^2)): Standard HRV metric
- HRV Index = RMSSD normalized to [0, 1] (1 = relaxed, 0 = stressed)
- Stress Index = 1 - HRV_Index (inverse relationship)

Heart Rate Variability (HRV) Interpretation:
- RMSSD > 50ms: High parasympathetic tone (relaxed)
- RMSSD 30-50ms: Normal (baseline on straight)
- RMSSD 15-30ms: Sympathetic activation (focused)
- RMSSD < 15ms: Extreme stress (cognitive saturation)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from collections import deque
import logging

try:
    import neurokit2 as nk
    NEUROKIT2_AVAILABLE = True
except ImportError:
    NEUROKIT2_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BioProcessor:
    """
    Real-time ECG signal processing and feature extraction.
    
    Processing Pipeline:
    1. Raw ECG → Cleaned ECG (neurokit2.ecg_clean)
    2. Cleaned ECG → R-peaks (neurokit2.ecg_peaks)
    3. R-peaks → RR intervals → HR, RMSSD
    4. HR + RMSSD → Stress Index
    """
    
    def __init__(
        self,
        sampling_rate: int = 500,
        window_size: int = 5,  # seconds
        rmssd_min: float = 8.0,  # milliseconds (panic threshold)
        rmssd_max: float = 60.0,  # milliseconds (relaxed threshold)
    ):
        """
        Initialize biometric processor.
        
        Args:
            sampling_rate: ECG sampling frequency (Hz)
            window_size: Processing window (seconds) for feature calculation
            rmssd_min: Minimum RMSSD (panic state)
            rmssd_max: Maximum RMSSD (relaxed state)
        """
        if not NEUROKIT2_AVAILABLE:
            raise ImportError("neurokit2 required: pip install neurokit2")
        
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.window_samples = window_size * sampling_rate
        
        # RMSSD normalization bounds
        self.rmssd_min = rmssd_min
        self.rmssd_max = rmssd_max
        
        # Buffers for streaming processing
        self.ecg_buffer = deque(maxlen=self.window_samples)
        self.rr_buffer = deque(maxlen=100)  # Store last 100 RR intervals
        
        # Feature storage
        self.last_hr = None
        self.last_rmssd = None
        self.last_stress_index = None
        
        logger.info(f"✓ BioProcessor initialized (fs={sampling_rate}Hz, window={window_size}s)")
    
    def clean_signal(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Clean ECG signal using neurokit2.ecg_clean().
        
        Removes:
        - Power line interference (50/60 Hz)
        - Baseline wander (low-frequency drift)
        - Movement artifacts
        
        Method: FIR band-pass filter (0.5-150 Hz) + baseline removal
        
        Args:
            ecg_signal: Raw ECG signal (n_samples,)
        
        Returns:
            ecg_clean: Cleaned ECG signal
        """
        try:
            # neurokit2.ecg_clean uses default 'neurokit' method:
            # - 0.5-150 Hz FIR band-pass
            # - Baseline removal using LOESS
            ecg_clean = nk.ecg_clean(
                ecg_signal,
                sampling_rate=self.sampling_rate,
                method='neurokit',
            )
            return ecg_clean
        except Exception as e:
            logger.error(f"Error in ECG cleaning: {e}")
            return ecg_signal  # Return raw on error
    
    def detect_peaks(self, ecg_clean: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Detect R-peaks (QRS complexes) from cleaned ECG.
        
        Uses neurokit2.ecg_peaks() with Neurokit method:
        - Continuous wavelet transform for peak detection
        - Adaptive threshold based on signal amplitude
        - QRS complex length: 80-120 ms
        
        Args:
            ecg_clean: Cleaned ECG signal (n_samples,)
        
        Returns:
            peaks: Array of R-peak sample indices
            info: Dictionary with peak information
        """
        try:
            _, peaks = nk.ecg_peaks(
                ecg_clean,
                sampling_rate=self.sampling_rate,
                method='neurokit',
                correct_artifacts=True,
            )
            return peaks['ECG_R_Peaks'], peaks
        except Exception as e:
            logger.error(f"Error in peak detection: {e}")
            return np.array([]), {}
    
    def compute_rr_intervals(self, peaks: np.ndarray) -> np.ndarray:
        """
        Compute RR intervals from R-peak locations.
        
        RR interval = time between consecutive R-peaks
        Unit: milliseconds
        
        Args:
            peaks: Array of R-peak sample indices
        
        Returns:
            rr_intervals: RR intervals in milliseconds (n_peaks-1,)
        """
        if len(peaks) < 2:
            return np.array([])
        
        # Time differences in samples
        rr_samples = np.diff(peaks)
        
        # Convert to milliseconds
        rr_intervals = (rr_samples / self.sampling_rate) * 1000
        
        return rr_intervals
    
    def compute_heart_rate(self, rr_intervals: np.ndarray) -> Optional[float]:
        """
        Compute instantaneous heart rate from RR intervals.
        
        Formula: HR = 60000 / mean(RR_interval_ms)
        Unit: beats per minute (bpm)
        
        Args:
            rr_intervals: RR intervals in milliseconds
        
        Returns:
            hr: Heart rate in bpm, or None if insufficient data
        """
        if len(rr_intervals) == 0:
            return None
        
        mean_rr = np.mean(rr_intervals)
        if mean_rr <= 0:
            return None
        
        hr = 60000.0 / mean_rr
        return np.clip(hr, 40, 200)  # Physiological bounds
    
    def compute_rmssd(self, rr_intervals: np.ndarray) -> Optional[float]:
        """
        Compute RMSSD (Heart Rate Variability).
        
        RMSSD = Root Mean Square of Successive Differences
        Formula: RMSSD = sqrt(mean((RR_i+1 - RR_i)^2))
        Unit: milliseconds
        
        Interpretation:
        - RMSSD > 50ms: High vagal tone (parasympathetic, relaxed)
        - RMSSD 30-50ms: Normal baseline
        - RMSSD 15-30ms: Sympathetic activation (stress)
        - RMSSD < 15ms: Cognitive saturation (danger zone)
        
        Args:
            rr_intervals: RR intervals in milliseconds (minimum 2)
        
        Returns:
            rmssd: RMSSD in milliseconds, or None if insufficient data
        """
        if len(rr_intervals) < 2:
            return None
        
        # Successive differences
        successive_diffs = np.diff(rr_intervals)
        
        # Root mean square
        rmssd = np.sqrt(np.mean(successive_diffs ** 2))
        
        return np.clip(rmssd, 0, 200)  # Physical bounds
    
    def compute_hvr_index(self, rmssd: Optional[float]) -> float:
        """
        Compute HRV Index (normalized RMSSD).
        
        Formula: HRV_Index = (RMSSD - RMSSD_min) / (RMSSD_max - RMSSD_min)
        Range: [0, 1]
        - 1.0: Fully relaxed (RMSSD = RMSSD_max)
        - 0.5: Normal state
        - 0.0: High stress (RMSSD = RMSSD_min)
        
        Args:
            rmssd: RMSSD in milliseconds
        
        Returns:
            hrv_index: Normalized HRV in [0, 1]
        """
        if rmssd is None:
            return 0.5  # Default to neutral
        
        rmssd_clipped = np.clip(rmssd, self.rmssd_min, self.rmssd_max)
        hrv_index = (rmssd_clipped - self.rmssd_min) / (self.rmssd_max - self.rmssd_min)
        
        return np.clip(hrv_index, 0, 1)
    
    def compute_stress_index(
        self,
        hr: Optional[float],
        hrv_index: float,
    ) -> float:
        """
        Compute composite Stress Index.
        
        Formula: Stress = (1 - HRV_Index) * alpha + HR_factor * (1 - alpha)
        - Emphasizes HRV collapse (parasympathetic withdrawal)
        - Incorporates HR elevation
        
        Range: [0, 1]
        - 0.0: Completely relaxed
        - 0.5: Moderate stress
        - 1.0: Maximum stress (cognitive saturation)
        
        Args:
            hr: Heart rate in bpm
            hrv_index: Normalized HRV [0, 1]
        
        Returns:
            stress_index: Stress level [0, 1]
        """
        # HRV component (70% weight)
        hrv_stress = 1.0 - hrv_index
        
        # HR component (30% weight)
        if hr is not None:
            # Normalize HR to [0, 1]: baseline=110 → 0, peak=180 → 1
            hr_stress = np.clip((hr - 110) / 70, 0, 1)
        else:
            hr_stress = 0.5
        
        # Composite stress
        stress_index = 0.7 * hrv_stress + 0.3 * hr_stress
        
        return np.clip(stress_index, 0, 1)
    
    def process_segment(
        self,
        ecg_signal: np.ndarray,
        return_detailed: bool = False,
    ) -> Dict[str, float]:
        """
        Process ECG segment and extract features.
        
        Complete pipeline:
        1. Clean ECG
        2. Detect R-peaks
        3. Compute RR intervals
        4. Extract HR, RMSSD
        5. Compute stress index
        
        Args:
            ecg_signal: Raw ECG signal (n_samples,)
            return_detailed: If True, return intermediate data
        
        Returns:
            dict with keys:
                - 'hr': Heart rate (bpm)
                - 'rmssd': Heart rate variability (ms)
                - 'hrv_index': Normalized HRV [0, 1]
                - 'stress_index': Stress level [0, 1]
                - 'n_peaks': Number of R-peaks detected
                - (optional) 'ecg_clean': Cleaned signal
                - (optional) 'peaks': R-peak indices
        """
        # Step 1: Clean
        ecg_clean = self.clean_signal(ecg_signal)
        
        # Step 2: Detect peaks
        peaks, peak_info = self.detect_peaks(ecg_clean)
        
        # Step 3: Compute RR intervals
        rr_intervals = self.compute_rr_intervals(peaks)
        
        # Step 4: Feature extraction
        hr = self.compute_heart_rate(rr_intervals)
        rmssd = self.compute_rmssd(rr_intervals)
        hrv_index = self.compute_hvr_index(rmssd)
        
        # Step 5: Stress computation
        stress_index = self.compute_stress_index(hr, hrv_index)
        
        # Store for streaming
        self.last_hr = hr
        self.last_rmssd = rmssd
        self.last_stress_index = stress_index
        
        # Update RR buffer
        self.rr_buffer.extend(rr_intervals)
        
        # Result
        result = {
            'hr': hr,
            'rmssd': rmssd,
            'hrv_index': hrv_index,
            'stress_index': stress_index,
            'n_peaks': len(peaks),
        }
        
        if return_detailed:
            result.update({
                'ecg_clean': ecg_clean,
                'peaks': peaks,
                'rr_intervals': rr_intervals,
            })
        
        return result
    
    def get_state_vector(self) -> np.ndarray:
        """
        Get current state as feature vector for RL observation.
        
        Returns:
            state: [hr_normalized, rmssd_index, stress_index]
        """
        if self.last_hr is None:
            # Default neutral state
            return np.array([0.5, 0.5, 0.5], dtype=np.float32)
        
        # Normalize HR to [0, 1]: 50 → 0, 200 → 1
        hr_norm = np.clip((self.last_hr - 50) / 150, 0, 1)
        
        return np.array([
            hr_norm,
            self.last_rmssd if self.last_rmssd is not None else 30.0,
            self.last_stress_index if self.last_stress_index is not None else 0.5,
        ], dtype=np.float32)
    
    def batch_process(
        self,
        ecg_signal: np.ndarray,
        overlap: float = 0.5,
    ) -> pd.DataFrame:
        """
        Process ECG in overlapping windows (batch mode).
        
        Args:
            ecg_signal: Full ECG signal (n_samples,)
            overlap: Window overlap factor [0, 1)
        
        Returns:
            DataFrame with features for each window
        """
        step_samples = int(self.window_samples * (1 - overlap))
        results = []
        
        for i in range(0, len(ecg_signal) - self.window_samples, step_samples):
            segment = ecg_signal[i:i + self.window_samples]
            features = self.process_segment(segment)
            
            time_start = i / self.sampling_rate
            time_end = (i + self.window_samples) / self.sampling_rate
            
            features['time_start'] = time_start
            features['time_end'] = time_end
            features['time_mid'] = (time_start + time_end) / 2
            
            results.append(features)
        
        return pd.DataFrame(results)


def analyze_ecg_episode(
    ecg_signal: np.ndarray,
    timestamps: np.ndarray,
    stress_ground_truth: np.ndarray,
    sampling_rate: int = 500,
) -> pd.DataFrame:
    """
    Complete ECG analysis for an episode.
    
    Args:
        ecg_signal: Full ECG signal
        timestamps: Time of each sample
        stress_ground_truth: Known stress levels for validation
        sampling_rate: Sampling rate (Hz)
    
    Returns:
        Analysis results DataFrame
    """
    processor = BioProcessor(sampling_rate=sampling_rate, window_size=5)
    
    results = processor.batch_process(ecg_signal, overlap=0.5)
    
    # Map stress ground truth to result times
    if len(stress_ground_truth) == len(ecg_signal):
        # Interpolate ground truth stress to match results
        results['stress_true'] = [
            stress_ground_truth[int(t * sampling_rate):int((t + 5) * sampling_rate)].mean()
            for t in results['time_start']
        ]
    
    return results


if __name__ == '__main__':
    # Demo
    logger.info("=" * 60)
    logger.info("BioProcessor Demo")
    logger.info("=" * 60)
    
    # Import bio_sim for data generation
    from bio_sim import BiometricDataSimulator, create_synthetic_telemetry
    
    # Create simulator and generate data
    logger.info("\n1. Generating synthetic ECG...")
    sim = BiometricDataSimulator(sampling_rate=500, seed=42)
    telemetry = create_synthetic_telemetry(duration=30)
    ecg_signal, timestamps, stress_profile = sim.generate_episode(telemetry, duration=30)
    
    # Process ECG
    logger.info("\n2. Processing ECG signal...")
    processor = BioProcessor(sampling_rate=500, window_size=5)
    results = processor.batch_process(ecg_signal, overlap=0.5)
    
    logger.info(f"\n3. Results (first 3 windows):")
    logger.info(f"\n{results[['time_mid', 'hr', 'rmssd', 'hrv_index', 'stress_index', 'n_peaks']].head(3).to_string()}")
    
    logger.info(f"\n4. Statistics:")
    logger.info(f"   HR range:           {results['hr'].min():.1f} - {results['hr'].max():.1f} bpm")
    logger.info(f"   RMSSD range:        {results['rmssd'].min():.1f} - {results['rmssd'].max():.1f} ms")
    logger.info(f"   Stress index range: {results['stress_index'].min():.2f} - {results['stress_index'].max():.2f}")
    
    logger.info("\n✓ Demo complete")
