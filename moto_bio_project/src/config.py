"""
Global Configuration and Hyperparameters for Bio-Adaptive Racing System
Controls simulation parameters, panic thresholds, and RL training settings
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


@dataclass
class SimulationConfig:
    """Physics and Telemetry Simulation Parameters"""
    
    # Circuit Parameters
    CIRCUIT_LENGTH_KM: float = 1.2
    NUM_LAPS: int = 100
    SAMPLING_RATE_TELEMETRY: int = 100  # Hz
    LAP_DURATION_SECONDS: int = 60
    
    # Speed Limits
    MAX_SPEED_KMH: float = 350.0
    MIN_SPEED_KMH: float = 0.0
    
    # Physics
    MAX_G_FORCE: float = 2.5
    MAX_LEAN_ANGLE: float = 65.0
    GRAVITY: float = 9.81
    
    # ECG/HRV Parameters
    ECG_SAMPLING_RATE: int = 500  # Hz (NeuroKit2 standard)
    RESTING_HR: int = 60  # bpm
    MAX_HR: int = 220  # Max heart rate formula: 220 - age
    
    # Stress Calculation
    PANIC_THRESHOLD: float = 0.8  # 0.0-1.0 normalized stress level
    HIGH_STRESS_THRESHOLD: float = 0.65
    MODERATE_STRESS_THRESHOLD: float = 0.40
    
    # Physiological Correlation (G-Force to Heart Rate)
    HR_BASELINE: int = 80  # bpm at rest
    HR_PER_G_FORCE: float = 25.0  # bpm increase per 1G
    HR_RESPONSE_LAG: float = 2.0  # seconds (exponential smoothing)


@dataclass
class RewardConfig:
    """Multi-Objective Reward Function Parameters (from Paper)"""
    
    # Reward Weights: Total = 0.50*speed + 0.35*safety - 0.15*stress²
    SPEED_WEIGHT: float = 0.50
    SAFETY_WEIGHT: float = 0.35
    STRESS_PENALTY_WEIGHT: float = 0.15
    
    # Speed Reward Scaling (0.0 to MAX_SPEED_KMH)
    SPEED_NORMALIZATION_FACTOR: float = 350.0  # Normalize to 0-1
    
    # Safety Reward: Minimize off-track penalties
    OFF_TRACK_PENALTY: float = -100.0
    LEAN_ANGLE_SAFE_MAX: float = 55.0
    LEAN_ANGLE_PENALTY_FACTOR: float = 5.0
    
    # Termination Conditions
    OFF_TRACK_THRESHOLD: float = 1.0  # road width in meters
    EPISODE_STEPS_MAX: int = 5000


@dataclass
class TrainingConfig:
    """PPO (Stable-Baselines3) Training Hyperparameters"""
    
    # Training Duration
    TOTAL_TIMESTEPS: int = 100000
    LEARNING_RATE: float = 3e-4
    
    # PPO Hyperparameters
    N_STEPS: int = 2048  # Batch size before update
    BATCH_SIZE: int = 64
    N_EPOCHS: int = 10  # Number of epochs for mini-batch
    GAMMA: float = 0.99  # Discount factor
    GAE_LAMBDA: float = 0.95  # Generalized Advantage Estimation
    CLIP_RANGE: float = 0.2  # PPO clip parameter
    ENTROPY_COEF: float = 0.0  # Entropy coefficient
    
    # Network Architecture
    POLICY_NETWORK_LAYERS: Tuple[int, ...] = (256, 256)
    ACTIVATION_FUNCTION: str = "relu"
    
    # Logging and Checkpoint
    LOG_INTERVAL: int = 10  # Log every N episodes
    SAVE_INTERVAL: int = 5  # Save checkpoint every N calls
    EVAL_EPISODES: int = 5


@dataclass
class VisualizationConfig:
    """Matplotlib Visualization Parameters"""
    
    # Figure Settings
    FIGURE_DPI: int = 300  # Publication quality
    FIGURE_SIZE: Tuple[float, float] = (16, 12)  # inches
    
    # Color Scheme
    COLOR_GREEN_CALM: str = "#2ecc71"  # Calm (stress < 0.40)
    COLOR_YELLOW_MODERATE: str = "#f39c12"  # Moderate (0.40-0.65)
    COLOR_RED_PANIC: str = "#e74c3c"  # Panic (stress > 0.65)
    
    # Font Settings
    FONT_SIZE_TITLE: int = 16
    FONT_SIZE_LABEL: int = 12
    FONT_SIZE_LEGEND: int = 10
    
    # Bio-Gate Visualization
    BIOGATE_MARKER_COLOR: str = "red"
    BIOGATE_MARKER_SIZE: int = 100
    BIOGATE_ALPHA: float = 0.7


@dataclass
class ProjectPaths:
    """File System Paths"""
    
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    SRC_DIR: Path = PROJECT_ROOT / "src"
    
    def __post_init__(self) -> None:
        """Create directories if they don't exist"""
        for path in [self.DATA_DIR, self.MODELS_DIR, self.LOGS_DIR]:
            path.mkdir(parents=True, exist_ok=True)


# Global Configuration Instances
SIM_CONFIG = SimulationConfig()
REWARD_CONFIG = RewardConfig()
TRAIN_CONFIG = TrainingConfig()
VIS_CONFIG = VisualizationConfig()
PATHS = ProjectPaths()


def get_config_summary() -> Dict[str, str]:
    """Return a summary of all configuration parameters"""
    return {
        "sim_panic_threshold": str(SIM_CONFIG.PANIC_THRESHOLD),
        "reward_weights": f"{REWARD_CONFIG.SPEED_WEIGHT}/{REWARD_CONFIG.SAFETY_WEIGHT}/{REWARD_CONFIG.STRESS_PENALTY_WEIGHT}",
        "training_total_steps": str(TRAIN_CONFIG.TOTAL_TIMESTEPS),
        "ppo_learning_rate": str(TRAIN_CONFIG.LEARNING_RATE),
        "figure_dpi": str(VIS_CONFIG.FIGURE_DPI),
    }
