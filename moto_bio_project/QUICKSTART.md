# 🏍️ Bio-Adaptive Haptic Coaching - QUICK START

## 30-Second Setup

### 1️⃣ Install Dependencies
```bash
pip install gymnasium stable-baselines3 neurokit2 pandas matplotlib numpy scikit-learn tensorboard
```

### 2️⃣ Choose Your Path

#### 🚀 **Option A: Quick Demo (5 minutes)**
```bash
cd moto_bio_project
python scripts/quick_demo.py
```
Generates sample results instantly for testing.

#### 🏁 **Option B: Full Production Run (10 minutes)**
```bash
cd moto_bio_project
python scripts/run_pipeline.py
```
Complete 4-phase pipeline with 100 laps of data and 100k training steps.

---

## 📊 What You Get

### Output Files
```
moto_bio_project/
├── data/
│   ├── telemetry.csv          # Racing data: speed, lean, HR, etc
│   ├── ecg_signal.npy         # Raw ECG signal (500 Hz)
│   └── hrv_metrics.json       # Heart rate variability metrics
│
├── models/
│   └── ppo_bio_adaptive.zip   # Trained RL model (ready to deploy)
│
└── logs/
    ├── bio_adaptive_results.png    # 📈 3-panel dashboard
    ├── training_metrics_plot.png   # 📊 Training statistics
    ├── evaluation_metrics.json     # 📋 Numerical results
    └── training_metrics.json       # 📉 Convergence data
```

### The 3-Panel Dashboard
The main output file **`bio_adaptive_results.png`** shows:

```
┌─────────────────────────────────────────────────────┐
│ 🟦 Speed (blue, left axis)                           │
│ 🟥 Lean Angle (red, right axis)                      │
│ Time progression →                                   │
├─────────────────────────────────────────────────────┤
│ 🟢 ECG Signal with Stress Zones                      │
│ 🟢 Green = Calm (0.0-0.4 stress)                    │
│ 🟡 Yellow = Moderate (0.4-0.65 stress)              │
│ 🔴 Red = Panic (0.65+ stress)                       │
├─────────────────────────────────────────────────────┤
│ 📳 Haptic Actions (4 levels + Bio-Gate markers)     │
│ 🟢 No Feedback | 🔵 Mild | 🟠 Warning | 🔴 Emergency│
│ 🔴 Red dashed = Bio-gate override (safety active)  │
└─────────────────────────────────────────────────────┘
```

---

## 🧠 The System Architecture

### 4 Phases

**Phase 1: Data Generation** (⏱️ 1 min)
- Simulates 100 laps of motorcycle racing
- Physics: speed, lean angle, G-forces
- Physiology: NeuroKit2 ECG synthesis
- Heart rate correlation with stress
- Output: CSV + ECG signal + HRV metrics

**Phase 2: Environment Setup** (⏱️ instant)
- Custom Gymnasium environment
- State: [Speed, Lean, G-Force, HRV, Stress]
- Actions: [No Feedback, Mild, Warning, Emergency]
- **Bio-Gating Override**: Safety constraint prevents RL from overwhelming drivers

**Phase 3: RL Training** (⏱️ 5-8 min)
- PPO algorithm (Stable-Baselines3)
- Reward: `0.50×speed + 0.35×safety - 0.15×stress²`
- Learns optimal haptic feedback timing
- Tracks bio-gate activation rate

**Phase 4: Evaluation** (⏱️ 1-2 min)
- Runs trained model on test data
- Generates publication-quality visualization
- Exports metrics and evaluation report

---

## 🎯 Key Concepts

### Bio-Gating (Safety Mechanism)
```
IF driver_stress > 0.80 THEN force_action = NO_FEEDBACK
```
- Non-learnable (cannot be bypassed)
- Firmware-level constraint
- Prevents information overload during panic
- Logged in output dashboard (red borders)

### Multi-Objective Reward
```
Reward = 0.50×speed + 0.35×safety - 0.15×stress²
```
- Speed: Encourages racing performance
- Safety: Minimizes off-track events
- Stress penalty: Avoids excessive cognitive load

### Stress Calculation
```
Stress = 0.6×(HR - RestingHR)/(MaxHR - RestingHR) 
         + 0.2×fatigue 
         + 0.2×off_track_penalty
```
Real physiological signals drive the coaching system!

---

## 🔧 Configuration

All parameters in `src/config.py`:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `NUM_LAPS` | 100 | Training data size |
| `PANIC_THRESHOLD` | 0.80 | Bio-gate activation level |
| `TOTAL_TIMESTEPS` | 100,000 | RL training duration |
| `LEARNING_RATE` | 3e-4 | PPO learning rate |
| `SPEED_WEIGHT` | 0.50 | Reward weight for speed |

Modify these for:
- ✏️ **Faster testing**: Reduce `NUM_LAPS` and `TOTAL_TIMESTEPS`
- ✏️ **Safer coaching**: Increase `PANIC_THRESHOLD` (e.g., 0.85)
- ✏️ **Speed focus**: Increase `SPEED_WEIGHT` (e.g., 0.60)

---

## 📚 Paper Integration

This implementation demonstrates all concepts from:
> **Bio-Cybernetic Adaptive Haptic Coaching for Competitive Motorcycle Racing**

Specific connections:
- **Section 4.1**: POMDP formulation → `MotoBioEnv` class
- **Section 4.2**: Bio-gating mechanism → `_bio_gating_mechanism()` method
- **Section 4.3**: Multi-objective reward → `_compute_reward()` function
- **Figure 4**: 3-panel dashboard → `bio_adaptive_results.png`

---

## 🚀 Advanced Usage

### Load Pre-Trained Model
```python
from stable_baselines3 import PPO

model = PPO.load("moto_bio_project/models/ppo_bio_adaptive")
action, _ = model.predict(observation)
```

### Custom Training Data
```python
from src.data_gen import SyntheticTelemetry

gen = SyntheticTelemetry()
session = gen.generate_race_session(n_laps=50)
telemetry = session.telemetry_df
```

### Modify Reward Function
Edit `src/environment.py`, method `_compute_reward()`:
```python
# Example: add lap time penalty
lap_time_penalty = 0.1 * current_lap_time
total_reward = speed_reward + safety_reward - stress_penalty - lap_time_penalty
```

---

## 📊 Monitor Training with TensorBoard
```bash
tensorboard --logdir=moto_bio_project/logs/
# Open http://localhost:6006
```

---

## ❓ FAQ

**Q: How long does full pipeline take?**
A: ~10 minutes on modern CPU. GPU optional (not required).

**Q: Can I run on my laptop?**
A: Yes! No special hardware needed. Memory requirement: ~2GB.

**Q: How do I use the trained model for real motorcycle data?**
A: Replace `src/data_gen.py` with your telemetry loader. Environment interface stays the same.

**Q: What does "bio-gate" mean?**
A: It's a safety valve. When driver stress peaks (detected via HR), the system stops sending feedback to avoid overload.

**Q: Can I modify the reward function?**
A: Yes! Edit `_compute_reward()` in `src/environment.py` and retrain.

---

## 📝 Citation

If you use this implementation, cite:

> Ruiz, R., et al. (2025). Bio-Adaptive Haptic Coaching for Competitive Motorcycle Racing. 
> https://github.com/rubences/Coaching-for-Competitive-Motorcycle-Racing

---

## 🎓 Learning Path

1. **Day 1**: Run quick demo → Read output dashboard
2. **Day 2**: Read `src/environment.py` → Understand the POMDP
3. **Day 3**: Modify `src/config.py` → Experiment with hyperparameters
4. **Day 4**: Retrain with custom data → Deploy to real system

---

## 📞 Support

- **Errors?** Check `logs/` for detailed training curves
- **Modifications?** All parameters in `src/config.py`
- **Questions?** Read docstrings in each `.py` file

---

**Ready to coach motorcycle racers with AI?** 🏍️⚡

```bash
python scripts/run_pipeline.py
```

Results in 10 minutes! 🚀
