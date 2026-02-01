# 🏍️ START HERE - Bio-Adaptive Haptic Coaching MLOps

## ⚡ 3-Minute Setup

### Step 1: Navigate & Install (1 minute)
```bash
cd moto_bio_project
pip install -r requirements.txt
```

### Step 2: Choose Your Path

#### Option A: Quick Demo (5 minutes total)
```bash
python scripts/quick_demo.py
```
**Good for**: Testing, understanding the pipeline, quick validation

#### Option B: Full Production (10 minutes total)
```bash
python scripts/run_pipeline.py
```
**Good for**: Publishing results, complete validation, getting metrics

### Step 3: View Results
```bash
# Main output (3-panel dashboard)
open logs/bio_adaptive_results.png

# Or check all outputs
ls -la logs/
```

---

## 📊 What You'll Get

```
logs/
├── bio_adaptive_results.png       ← 3-PANEL DASHBOARD (for paper!)
├── training_metrics_plot.png      ← Training convergence
├── evaluation_metrics.json        ← Numerical results
└── training_metrics.json          ← Training stats

models/
└── ppo_bio_adaptive.zip           ← Trained RL model

data/
├── telemetry.csv                  ← Racing data
├── ecg_signal.npy                 ← ECG signal
└── hrv_metrics.json               ← HRV metrics
```

---

## 🎯 Key Files

| What | Where |
|------|-------|
| **Quick start** | `moto_bio_project/QUICKSTART.md` |
| **Full documentation** | `moto_bio_project/README.md` |
| **Configuration** | `moto_bio_project/src/config.py` |
| **Code reference** | `moto_bio_project/src/` |
| **Execution** | `moto_bio_project/scripts/` |

---

## ⚙️ Customize (Optional)

Edit `moto_bio_project/src/config.py`:

```python
# For faster testing:
SIM_CONFIG.NUM_LAPS = 10              # (default: 100)
TRAIN_CONFIG.TOTAL_TIMESTEPS = 10000  # (default: 100000)

# For safer coaching:
SIM_CONFIG.PANIC_THRESHOLD = 0.75     # (default: 0.80)

# For speed emphasis:
REWARD_CONFIG.SPEED_WEIGHT = 0.70     # (default: 0.50)
```

Then re-run the pipeline.

---

## 🚨 Troubleshooting

**Q: ImportError for gymnasium/stable_baselines3?**
```bash
pip install gymnasium stable-baselines3 neurokit2
```

**Q: Need to see what's happening?**
```bash
# Monitor training in real-time
tensorboard --logdir=moto_bio_project/logs/
```

**Q: Want just the data without training?**
Edit `run_pipeline.py` or use the modules separately.

---

## 📈 System Architecture

```
Phase 1: DATA              Phase 2: ENV              Phase 3: TRAIN           Phase 4: VIZ
┌─────────────────┐      ┌─────────────────┐       ┌─────────────────┐      ┌──────────────┐
│ Physics Sim     │      │ Gymnasium POMDP │       │ PPO (100k steps)│      │ 3-Panel Plot │
│ + ECG Synthesis │  →   │ + Bio-Gating    │   →   │ + Callbacks     │  →   │ + Metrics    │
│ (100 laps)      │      │ (5D state)      │       │ (Stable-B3)     │      │ (300 DPI)    │
└─────────────────┘      └─────────────────┘       └─────────────────┘      └──────────────┘
                              IF stress > 0.8
                              force action = 0
```

---

## ✅ Validation Checklist

After running the pipeline, you should see:

- ✅ `data/telemetry.csv` created (~MB)
- ✅ `data/ecg_signal.npy` created
- ✅ `models/ppo_bio_adaptive.zip` created
- ✅ `logs/bio_adaptive_results.png` created (3-panel dashboard)
- ✅ `logs/training_metrics_plot.png` created
- ✅ Console output showing training progress

---

## 🎓 Paper Integration

The results PNG (`bio_adaptive_results.png`) is ready to use as:
- **Figure 4** in your research paper
- Publication-quality (300 DPI)
- Shows all 3 key aspects: dynamics, physiology, control

---

## 📞 Need Help?

| Topic | File |
|-------|------|
| 30-second guide | `moto_bio_project/QUICKSTART.md` |
| Complete reference | `moto_bio_project/README.md` |
| Navigation | `moto_bio_project/INDEX.md` |
| Configuration | `moto_bio_project/src/config.py` |

---

## 🚀 You're All Set!

Run this now:
```bash
cd moto_bio_project && python scripts/run_pipeline.py
```

Results in 10 minutes! 🎉

---

**Last Updated**: January 17, 2025  
**Status**: ✅ Ready to run
