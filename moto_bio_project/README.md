# Bio-Adaptive Haptic Coaching System - MLOps Implementation

## 📋 Overview

This is a **complete, production-ready** implementation of the "Bio-Adaptive Haptic Coaching for Competitive Motorcycle Racing" research system. The project integrates:

- **Synthetic Data Generation**: Physics-based motorcycle racing simulation with NeuroKit2 ECG generation
- **Custom Gymnasium Environment**: POMDP with bio-gating safety override mechanism
- **Reinforcement Learning**: PPO agent trained on multi-objective reward function
- **Publication-Quality Visualization**: 3-panel dashboard with ECG, stress zones, and haptic actions

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

```bash
python scripts/run_pipeline.py
```

This executes all 4 phases automatically:
- **Phase 1**: Generate 100 laps of synthetic telemetry + ECG
- **Phase 2**: Initialize MotoBioEnv with bio-gating
- **Phase 3**: Train PPO agent (100k timesteps)
- **Phase 4**: Evaluate and create visualizations

**Runtime**: ~5-10 minutes on modern hardware

### 3. View Results

All outputs saved to:
- `data/` - Generated telemetry and ECG signals
- `models/` - Trained PPO model (.zip)
- `logs/` - Training metrics, evaluation plots, and results PNG

## 📁 Project Structure

```
moto_bio_project/
├── src/
│   ├── config.py        # Global hyperparameters (panic thresholds, rewards, etc)
│   ├── data_gen.py      # Physics + NeuroKit2 ECG generation (SyntheticTelemetry class)
│   ├── environment.py   # Custom Gym Env with Bio-Gating (MotoBioEnv class)
│   ├── train.py         # PPO training with callbacks
│   └── visualize.py     # 3-panel publication-quality dashboard
├── scripts/
│   └── run_pipeline.py  # Master orchestrator (execute this!)
├── models/              # Saved PPO models
├── logs/                # Metrics, checkpoints, visualizations
├── data/                # Generated telemetry and ECG
├── requirements.txt     # Dependencies
└── README.md            # This file
```

## 🧬 Key Components

### 1. Synthetic Data Generation (`src/data_gen.py`)

**Class**: `SyntheticTelemetry`

Generates realistic motorcycle racing telemetry:
- **Circuit Physics**: 1.2 km circuit with 4 corners + 2 straights
- **Speed Profile**: Acceleration/deceleration with realistic corner speeds
- **G-Force Dynamics**: Lateral G from lean angle, longitudinal G from acceleration
- **Physiological Correlation**: Heart rate increases with G-force (exponential lag)
- **ECG Simulation**: NeuroKit2-based ECG synthesis at 500 Hz
- **HRV Metrics**: RMSSD, SDNN, LF/HF ratios (cognitive load proxies)
- **Vibration Noise**: Speed-dependent ECG artifacts for realism

**Output**:
```
data/
├── telemetry.csv          # 100 laps of [time, speed, lean, g_force, hr]
├── ecg_signal.npy         # Raw ECG at 500 Hz (5000+ samples)
├── hrv_metrics.json       # RMSSD, SDNN, LF/HF
└── metadata.json          # Circuit info, mean/max values
```

### 2. Custom Environment (`src/environment.py`)

**Class**: `MotoBioEnv(gym.Env)`

Implements the POMDP from the research paper:

**State Space** (5D continuous):
- Speed (0-350 km/h)
- Lean angle (0-65°)
- G-force (0-2.5G)
- HRV index (0-1, inverse of HR)
- Stress level (0-1, normalized)

**Action Space** (Discrete 4):
- 0 = No Feedback
- 1 = Mild Haptic (gentle vibration)
- 2 = Warning Haptic (moderate vibration)
- 3 = Emergency Haptic (strong vibration)

**Reward Function** (multi-objective, from paper):
```
R = 0.50 × speed + 0.35 × safety - 0.15 × stress²
```

**Bio-Gating Mechanism** (non-learnable safety):
```python
IF stress > 0.80 THEN force action = 0 (no feedback)
```
This is a firmware-level constraint that cannot be circumvented by the RL policy.

### 3. Training Module (`src/train.py`)

**Algorithm**: PPO (Proximal Policy Optimization) via Stable-Baselines3

**Hyperparameters**:
- Learning rate: 3e-4
- N steps per batch: 2048
- Batch size: 64
- Gamma (discount): 0.99
- GAE lambda: 0.95

**Callbacks**:
- Checkpoint save every 10k timesteps
- `BioAdaptiveCallback`: Tracks episode rewards and bio-gate events
- TensorBoard logging

**Output**:
```
models/
├── ppo_bio_adaptive.zip       # Best trained model
├── ppo_checkpoint_*.zip       # Intermediate checkpoints
logs/
├── training_metrics.json      # Mean/max/min rewards
└── events.out.tfevents...     # TensorBoard events
```

### 4. Visualization (`src/visualize.py`)

**Output**: 3-panel publication-quality PNG (300 DPI)

**Panel 1 - Speed & Lean Angle**:
- Time series of speed (0-350 km/h, blue)
- Time series of lean angle (0-65°, red)
- Grid background for readability

**Panel 2 - ECG with Stress Zones**:
- Raw ECG signal (black line)
- Background colors indicate driver stress:
  - 🟢 Green (0.0-0.4): Calm, optimal learning
  - 🟡 Yellow (0.4-0.65): Moderate stress
  - 🔴 Red (0.65-1.0): High panic, need cooling down

**Panel 3 - Haptic Actions**:
- Timeline of 4 action levels (colors)
- 🔴 Red dashed borders show bio-gate overrides
- Indicates when the safety mechanism prevented over-stimulation

## 🔧 Configuration (`src/config.py`)

All hyperparameters are centralized in dataclasses:

```python
SIM_CONFIG.PANIC_THRESHOLD = 0.8        # Stress level for bio-gate activation
REWARD_CONFIG.SPEED_WEIGHT = 0.50       # Speed contribution to reward
TRAIN_CONFIG.TOTAL_TIMESTEPS = 100000   # Training duration
VIS_CONFIG.FIGURE_DPI = 300             # Publication quality
```

Modify these to experiment with different configurations!

## 📊 Expected Results

After running the pipeline, you should see:

1. **Training Convergence**: Reward increases over episodes
2. **Bio-Gate Effectiveness**: 5-15% activation rate (preventing panic)
3. **Off-Track Reduction**: Decreases as agent learns to corner safely
4. **Mean Episode Reward**: Converges to ~200+ (from baseline ~100)

## 🎓 Research Paper Integration

This implementation demonstrates the concepts from:

**Bio-Cybernetic Adaptive Haptic Coaching for Competitive Motorcycle Racing**

Key paper sections:
- **Section 4.1 (POMDP)**: Implemented in `MotoBioEnv`
- **Section 4.2 (Bio-Gating)**: `_bio_gating_mechanism()` method
- **Section 4.3 (Reward)**: `_compute_reward()` function
- **Section 5 (Results)**: Generated via `src/visualize.py`

## 🔬 Advanced Usage

### Custom Training Configuration

Edit `src/config.py`:

```python
TRAIN_CONFIG.TOTAL_TIMESTEPS = 200000  # Longer training
TRAIN_CONFIG.LEARNING_RATE = 1e-4      # Lower learning rate
SIM_CONFIG.NUM_LAPS = 200              # More training data
```

### Load and Evaluate Existing Model

```python
from src.environment import MotoBioEnv
from stable_baselines3 import PPO

env = MotoBioEnv()
model = PPO.load("models/ppo_bio_adaptive")

obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### Generate New Training Data

```python
from src.data_gen import SyntheticTelemetry

gen = SyntheticTelemetry()
session = gen.generate_race_session(n_laps=50)
# Now use session.telemetry_df in MotoBioEnv
```

## 📈 Monitoring Training

Real-time monitoring with TensorBoard:

```bash
tensorboard --logdir=logs/
```

Then open `http://localhost:6006` in your browser.

## 🚨 Troubleshooting

**ImportError: No module named 'gymnasium'**
```bash
pip install gymnasium
```

**Memory issues with large datasets**:
Reduce `SIM_CONFIG.NUM_LAPS` in `src/config.py`

**Training is slow**:
Reduce `TRAIN_CONFIG.TOTAL_TIMESTEPS` for quick testing

## 📝 License

Same as parent project (Coaching-for-Competitive-Motorcycle-Racing)

## 👨‍🔬 Authors

Bio-Adaptive Racing Team | 2025

---

**Ready to deploy?** Run `python scripts/run_pipeline.py` and enjoy publication-ready results! 🏍️
