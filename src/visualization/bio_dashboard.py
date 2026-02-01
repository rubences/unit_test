"""
Multimodal Biometric Dashboard: Visualization of ECG, Telemetry, and Stress Integration

Three-panel visualization:
1. TOP:    Motorcycle Telemetry (Speed, G-Force, Lean Angle)
2. MIDDLE: ECG Signal (Raw vs. Cleaned) with R-Peak Markers
3. BOTTOM: Stress Evolution (RMSSD Index, HR, Cognitive Load) with danger zones

Dashboard Purpose:
- Real-time visualization during training
- Post-episode analysis of pilot stress response
- Validation of Panic Freeze events
- Research/publication figures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import Optional, Tuple, Dict
import logging

try:
    import neurokit2 as nk
    NEUROKIT2_AVAILABLE = True
except ImportError:
    NEUROKIT2_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BiometricDashboard:
    """
    Multimodal visualization dashboard for motorcycle + biometrics integration.
    
    Three synchronized plots:
    1. Telemetry: Speed, G-Force, Lean Angle
    2. ECG: Raw signal, cleaned signal, R-peak markers
    3. Stress: RMSSD index, HR, stress level, panic zones
    """
    
    def __init__(self, figsize: Tuple[int, int] = (16, 10)):
        """
        Initialize dashboard.
        
        Args:
            figsize: Figure size (inches)
        """
        self.figsize = figsize
        self.fig = None
        self.axes = None
        logger.info("✓ BiometricDashboard initialized")
    
    def create_figure(self) -> Tuple[plt.Figure, list]:
        """
        Create figure with 3 aligned subplots.
        
        Returns:
            fig, axes
        """
        self.fig = plt.figure(figsize=self.figsize, constrained_layout=True)
        gs = GridSpec(3, 1, figure=self.fig, height_ratios=[1, 1.2, 1])
        
        ax_telemetry = self.fig.add_subplot(gs[0])
        ax_ecg = self.fig.add_subplot(gs[1], sharex=ax_telemetry)
        ax_stress = self.fig.add_subplot(gs[2], sharex=ax_telemetry)
        
        self.axes = [ax_telemetry, ax_ecg, ax_stress]
        
        # Configure axes
        ax_telemetry.set_ylabel('Telemetry', fontsize=11, fontweight='bold')
        ax_ecg.set_ylabel('ECG Signal (mV)', fontsize=11, fontweight='bold')
        ax_stress.set_ylabel('Stress Index', fontsize=11, fontweight='bold')
        ax_stress.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
        
        return self.fig, self.axes
    
    def plot_telemetry(
        self,
        ax: plt.Axes,
        timestamps: np.ndarray,
        speed: np.ndarray,
        g_force: np.ndarray,
        lean_angle: np.ndarray,
    ) -> None:
        """
        Plot motorcycle telemetry on top panel.
        
        Args:
            ax: Matplotlib axis
            timestamps: Time array (seconds)
            speed: Speed array (m/s)
            g_force: G-force array
            lean_angle: Lean angle array (degrees)
        """
        # Secondary axis for lean angle
        ax2 = ax.twinx()
        
        # Plot speed and G-force
        ax.plot(timestamps, speed, 'b-', linewidth=2, label='Speed (m/s)')
        ax.plot(timestamps, g_force * 20, 'r-', linewidth=2, label='G-Force (x20)')
        
        # Plot lean angle (different color)
        ax2.plot(timestamps, lean_angle, 'g--', linewidth=1.5, alpha=0.7, label='Lean Angle (°)')
        
        # Formatting
        ax.set_ylim([0, max(speed.max(), g_force.max() * 20) * 1.1])
        ax2.set_ylim([0, 70])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=9)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.set_ylabel('Lean Angle (°)', fontsize=10)
    
    def plot_ecg(
        self,
        ax: plt.Axes,
        timestamps: np.ndarray,
        ecg_raw: np.ndarray,
        ecg_clean: Optional[np.ndarray] = None,
        peaks: Optional[np.ndarray] = None,
        sampling_rate: int = 500,
    ) -> None:
        """
        Plot ECG signal with preprocessing and peak markers.
        
        Args:
            ax: Matplotlib axis
            timestamps: Time array (seconds)
            ecg_raw: Raw ECG signal
            ecg_clean: Cleaned ECG (optional)
            peaks: R-peak indices (optional)
            sampling_rate: Sampling rate (Hz)
        """
        # Plot raw signal (light gray)
        ax.plot(timestamps, ecg_raw, 'gray', linewidth=0.5, alpha=0.5, label='Raw ECG')
        
        # Plot cleaned signal
        if ecg_clean is not None:
            ax.plot(timestamps, ecg_clean, 'b-', linewidth=1.5, label='Cleaned ECG')
        
        # Mark R-peaks
        if peaks is not None and len(peaks) > 0:
            peak_times = peaks / sampling_rate
            peak_values = ecg_clean[peaks] if ecg_clean is not None else ecg_raw[peaks]
            
            ax.scatter(peak_times, peak_values, color='red', s=50, marker='^',
                      label=f'R-Peaks (n={len(peaks)})', zorder=5)
        
        # Formatting
        ax.set_ylim([ecg_raw.min() * 1.2, ecg_raw.max() * 1.2])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=9)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    def plot_stress(
        self,
        ax: plt.Axes,
        timestamps: np.ndarray,
        hrv_index: np.ndarray,
        stress_index: np.ndarray,
        panic_zones: Optional[np.ndarray] = None,
    ) -> None:
        """
        Plot stress evolution with danger zones.
        
        Args:
            ax: Matplotlib axis
            timestamps: Time array (seconds)
            hrv_index: HRV normalized index (0=stressed, 1=relaxed)
            stress_index: Stress index (0=relaxed, 1=panicked)
            panic_zones: Boolean array indicating panic regions (optional)
        """
        # Plot HRV index (inverse = stress)
        ax.plot(timestamps, 1 - hrv_index, 'b-', linewidth=2, label='HRV-based Stress')
        
        # Plot stress index (direct)
        ax.plot(timestamps, stress_index, 'r--', linewidth=1.5, alpha=0.7, label='Composite Stress')
        
        # Fill danger zones (RMSSD collapse)
        if panic_zones is not None:
            # Find continuous panic regions
            panic_diff = np.diff(np.concatenate(([False], panic_zones, [False])).astype(int))
            starts = np.where(panic_diff == 1)[0]
            ends = np.where(panic_diff == -1)[0]
            
            for start, end in zip(starts, ends):
                ax.axvspan(timestamps[min(start, len(timestamps)-1)],
                          timestamps[min(end-1, len(timestamps)-1)],
                          alpha=0.2, color='red', label='Panic Zone' if start == starts[0] else '')
        
        # Danger thresholds
        ax.axhline(y=0.7, color='orange', linestyle=':', linewidth=1.5, label='High Stress Threshold')
        ax.axhline(y=0.9, color='red', linestyle=':', linewidth=1.5, label='Panic Threshold')
        
        # Formatting
        ax.set_ylim([0, 1])
        ax.set_xlim([timestamps[0], timestamps[-1]])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
    
    def plot_episode(
        self,
        timestamps: np.ndarray,
        ecg_signal: np.ndarray,
        telemetry_df: pd.DataFrame,
        ecg_features_df: pd.DataFrame,
        sampling_rate: int = 500,
        title: str = 'Motorcycle + Biometrics Integration',
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create complete multimodal visualization.
        
        Args:
            timestamps: Time array (seconds)
            ecg_signal: Raw ECG signal
            telemetry_df: DataFrame with [timestamp, g_force, lean_angle, speed]
            ecg_features_df: DataFrame with processed features [hrv_index, stress_index, etc.]
            sampling_rate: ECG sampling rate (Hz)
            title: Figure title
            save_path: Path to save figure (optional)
        
        Returns:
            fig
        """
        # Create figure
        self.create_figure()
        ax_telemetry, ax_ecg, ax_stress = self.axes
        
        # Interpolate telemetry to ECG time grid
        if 'timestamp' in telemetry_df.columns:
            # Align to ECG time grid
            speed_interp = np.interp(
                timestamps,
                telemetry_df['timestamp'],
                telemetry_df['speed'],
                left=telemetry_df['speed'].iloc[0],
                right=telemetry_df['speed'].iloc[-1],
            )
            g_force_interp = np.interp(
                timestamps,
                telemetry_df['timestamp'],
                telemetry_df['g_force'],
                left=0,
                right=telemetry_df['g_force'].iloc[-1],
            )
            lean_interp = np.interp(
                timestamps,
                telemetry_df['timestamp'],
                telemetry_df['lean_angle'],
                left=0,
                right=telemetry_df['lean_angle'].iloc[-1],
            )
        else:
            # Use indices
            speed_interp = telemetry_df['speed'].values
            g_force_interp = telemetry_df['g_force'].values
            lean_interp = telemetry_df['lean_angle'].values
        
        # Plot telemetry
        self.plot_telemetry(
            ax_telemetry,
            timestamps,
            speed_interp,
            g_force_interp,
            lean_interp,
        )
        
        # Process ECG for visualization
        if NEUROKIT2_AVAILABLE:
            ecg_clean = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate, method='neurokit')
            _, peaks = nk.ecg_peaks(ecg_clean, sampling_rate=sampling_rate, method='neurokit')
            peak_indices = peaks['ECG_R_Peaks']
        else:
            ecg_clean = None
            peak_indices = None
        
        # Plot ECG
        self.plot_ecg(
            ax_ecg,
            timestamps,
            ecg_signal,
            ecg_clean=ecg_clean,
            peaks=peak_indices,
            sampling_rate=sampling_rate,
        )
        
        # Interpolate stress features to ECG time grid
        if 'time_mid' in ecg_features_df.columns:
            hrv_idx_interp = np.interp(
                timestamps,
                ecg_features_df['time_mid'],
                ecg_features_df['hrv_index'],
                left=ecg_features_df['hrv_index'].iloc[0],
                right=ecg_features_df['hrv_index'].iloc[-1],
            )
            stress_idx_interp = np.interp(
                timestamps,
                ecg_features_df['time_mid'],
                ecg_features_df['stress_index'],
                left=ecg_features_df['stress_index'].iloc[0],
                right=ecg_features_df['stress_index'].iloc[-1],
            )
        else:
            hrv_idx_interp = ecg_features_df['hrv_index'].values
            stress_idx_interp = ecg_features_df['stress_index'].values
        
        # Detect panic zones (RMSSD < 15ms + high stress)
        if 'rmssd' in ecg_features_df.columns:
            rmssd_interp = np.interp(
                timestamps,
                ecg_features_df['time_mid'],
                ecg_features_df['rmssd'],
                left=ecg_features_df['rmssd'].iloc[0],
                right=ecg_features_df['rmssd'].iloc[-1],
            )
            panic_mask = rmssd_interp < 15.0
        else:
            panic_mask = stress_idx_interp > 0.7
        
        # Plot stress
        self.plot_stress(
            ax_stress,
            timestamps,
            hrv_idx_interp,
            stress_idx_interp,
            panic_zones=panic_mask,
        )
        
        # Global title and formatting
        self.fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        
        # Save if requested
        if save_path:
            self.fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Dashboard saved to {save_path}")
        
        return self.fig


def create_analysis_report(
    timestamps: np.ndarray,
    ecg_signal: np.ndarray,
    telemetry_df: pd.DataFrame,
    ecg_features_df: pd.DataFrame,
    sampling_rate: int = 500,
) -> Dict[str, float]:
    """
    Generate statistical summary of biometric episode.
    
    Args:
        timestamps: Time array
        ecg_signal: ECG signal
        telemetry_df: Telemetry DataFrame
        ecg_features_df: Features DataFrame
        sampling_rate: Sampling rate (Hz)
    
    Returns:
        Statistics dictionary
    """
    stats = {
        'duration_seconds': timestamps[-1],
        'mean_speed_ms': telemetry_df['speed'].mean(),
        'max_g_force': telemetry_df['g_force'].max(),
        'mean_lean_angle': telemetry_df['lean_angle'].mean(),
        'mean_hr_bpm': ecg_features_df['hr'].mean(),
        'min_rmssd_ms': ecg_features_df['rmssd'].min(),
        'max_stress_index': ecg_features_df['stress_index'].max(),
        'mean_stress_index': ecg_features_df['stress_index'].mean(),
        'time_in_high_stress': (ecg_features_df['stress_index'] > 0.7).sum() / len(ecg_features_df),
    }
    
    return stats


if __name__ == '__main__':
    # Demo
    logger.info("=" * 60)
    logger.info("BiometricDashboard Demo")
    logger.info("=" * 60)
    
    # Generate synthetic data
    from src.data.bio_sim import BiometricDataSimulator, create_synthetic_telemetry
    from src.features.bio_processor import BioProcessor
    
    logger.info("\n1. Generating synthetic data...")
    sim = BiometricDataSimulator(sampling_rate=500, seed=42)
    telemetry = create_synthetic_telemetry(duration=60)
    ecg_signal, timestamps, stress_profile = sim.generate_episode(telemetry, duration=60)
    
    logger.info("\n2. Processing ECG...")
    processor = BioProcessor(sampling_rate=500, window_size=5)
    ecg_features = processor.batch_process(ecg_signal, overlap=0.5)
    
    logger.info("\n3. Creating dashboard...")
    dashboard = BiometricDashboard(figsize=(16, 10))
    fig = dashboard.plot_episode(
        timestamps=timestamps,
        ecg_signal=ecg_signal,
        telemetry_df=telemetry,
        ecg_features_df=ecg_features,
        sampling_rate=500,
        title='Motorcycle Racing + Biometric Stress Integration',
        save_path='/tmp/biometric_dashboard.png',
    )
    
    logger.info("\n4. Statistics:")
    stats = create_analysis_report(timestamps, ecg_signal, telemetry, ecg_features)
    for key, value in stats.items():
        if isinstance(value, float):
            logger.info(f"   {key:30s}: {value:10.2f}")
        else:
            logger.info(f"   {key:30s}: {value}")
    
    logger.info("\n✓ Demo complete - figure saved to /tmp/biometric_dashboard.png")
    
    # Show figure
    plt.show()
