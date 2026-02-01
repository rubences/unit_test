# Moto-Edge-RL: Technical Implementation Guide

This guide provides a comprehensive, step-by-step technical deep dive into implementing the Moto-Edge-RL system.

## Table of Contents

1. [Phase 1: Data Acquisition and Hardware](#phase-1-data-acquisition-and-hardware)
2. [Phase 2: Simulation Environment Design](#phase-2-simulation-environment-design)
3. [Phase 3: Multi-Agent Architecture](#phase-3-multi-agent-architecture)
4. [Phase 4: Offline RL Training](#phase-4-offline-rl-training)
5. [Phase 5: Edge AI Deployment](#phase-5-edge-ai-deployment)

---

## Phase 1: Data Acquisition and Hardware

The first step is to instrument the motorcycle and rider. We don't rely on GPS (imprecise in corners) or cameras (slow), but on pure inertia.

### Hardware Setup

**Sensor: BNO055 9-Axis IMU**
- Mounted on helmet or chassis
- Configured to sample at 100Hz
- Provides:
  - 3-axis accelerometer
  - 3-axis gyroscope
  - 3-axis magnetometer

**Microcontroller Options**
- **ESP32**: Primary target with WiFi/BLE for data streaming
- **Arduino Nicla Sense ME**: Alternative with integrated IMU

### Data Preprocessing

```python
import numpy as np
from scipy.signal import butter, filtfilt

def kalman_filter(raw_data, process_noise=0.01, measurement_noise=0.1):
    """
    Extended Kalman Filter to remove motor vibration noise
    """
    # Implementation details
    pass

def preprocess_imu_data(acceleration, gyro, magnetometer):
    """
    Apply filtering and sensor fusion
    """
    # Butterworth low-pass filter to remove high-frequency noise
    b, a = butter(4, 0.1, btype='low')
    filtered_accel = filtfilt(b, a, acceleration, axis=0)
    
    return filtered_accel, gyro, magnetometer
```

### Dataset Creation

Recording sessions from professional and amateur riders are converted to the **Minari** standard format for offline RL:

```python
import minari
import gymnasium as gym

# Create a Minari dataset
dataset = minari.create_dataset_from_buffers(
    dataset_id="racing_dataset-v0",
    env=gym.make("MotorcycleRacing-v0"),
    buffer=recorded_episodes,
    algorithm_name="behavior_cloning",
    author="Moto-Edge-RL Team",
)

# Save to data/processed/racing_dataset.hdf5
dataset.save()
```

**Dataset Structure:**
```
data/
├── raw/
│   ├── pro_rider_session_001.csv
│   ├── pro_rider_session_002.csv
│   └── amateur_rider_session_001.csv
└── processed/
    └── racing_dataset.hdf5  # Minari format
```

---

## Phase 2: Simulation Environment Design

Before touching the track, the AI must learn physics. We create a custom Gymnasium environment.

### Environment Implementation

The `motorcycle_env.py` file (located in `/simulation`) implements `MotorcycleEnv-v0`:

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MotorcycleEnv(gym.Env):
    """
    Custom Gymnasium environment for motorcycle racing.
    
    Observation Space (8-dimensional):
        - velocity (m/s): [0, 100]
        - roll_angle (degrees): [-60, 60]
        - lateral_g (G): [-2.0, 2.0]
        - distance_to_apex (m): [0, 500]
        - throttle_position: [0, 1]
        - brake_pressure: [0, 1]
        - racing_line_deviation (m): [-10, 10]
        - tire_friction_usage: [0, 1]
    
    Action Space (3-dimensional):
        - haptic_left_intensity: [0, 1]
        - haptic_right_intensity: [0, 1]
        - haptic_frequency (Hz): [50, 300]
    """
    
    def __init__(self, track_name="silverstone"):
        super().__init__()
        # See full implementation in simulation/motorcycle_env.py
        pass
```

### Physics Models

**Kamm Circle (Traction Circle)**

The Kamm Circle represents the tire's total friction budget. Longitudinal (braking/acceleration) and lateral (cornering) forces must fit within the circle:

```
sqrt(F_lateral² + F_longitudinal²) ≤ μ * F_normal
```

Implementation:

```python
def calculate_kamm_circle_violation(lateral_g, brake_pressure):
    """
    Returns friction usage ratio.
    >1.0 means exceeding tire limits (crash risk).
    """
    TIRE_FRICTION_COEFF = 1.5  # Dry tarmac
    
    lateral_force = abs(lateral_g) / TIRE_FRICTION_COEFF
    longitudinal_force = brake_pressure
    
    friction_usage = np.sqrt(lateral_force**2 + longitudinal_force**2)
    return friction_usage
```

### Multi-Objective Reward Function

Using **MO-Gymnasium** for vectorized rewards:

```python
def calculate_reward(state, action, next_state):
    """
    Returns: (r_speed, r_safety, r_smooth)
    """
    velocity = next_state[0]
    friction_usage = next_state[7]
    
    # Speed reward: Encourage high exit velocity
    r_speed = velocity / MAX_VELOCITY
    
    # Safety reward: Massive penalty for exceeding grip
    if friction_usage > 1.0:
        r_safety = -10.0 * (friction_usage - 1.0)
    else:
        r_safety = 0.1 * (1.0 - friction_usage)
    
    # Smoothness reward: Penalize abrupt haptic changes
    haptic_change = np.linalg.norm(action - last_action)
    r_smooth = -0.5 * haptic_change
    
    # Weighted combination
    total = 1.0*r_speed + 2.0*r_safety + 0.5*r_smooth
    
    return r_speed, r_safety, r_smooth
```

---

## Phase 3: Multi-Agent Architecture

This is where "agentic" intelligence resides. We use **PettingZoo** to coordinate two cooperative agents.

### Agent Definitions

**1. Physics Agent (Trajectory Planner)**
- **Input**: Telemetry (velocity, roll, G-forces)
- **Output**: Deviation from optimal racing line (continuous value)
- **Role**: Pure mathematical calculation of trajectory error

**2. Coach Agent (Haptic Controller)**
- **Input**: Physics Agent output + Rider stress indicators
- **Output**: Haptic action (left, right, frequency)
- **Role**: Decides when and how to communicate feedback

### PettingZoo Implementation

```python
from pettingzoo import ParallelEnv
from gymnasium import spaces

class MultiAgentRacingEnv(ParallelEnv):
    """
    Two cooperative agents:
    - 'physics_agent': Calculates trajectory error
    - 'coach_agent': Decides haptic feedback
    """
    
    def __init__(self):
        self.possible_agents = ["physics_agent", "coach_agent"]
        self.agents = self.possible_agents[:]
        
        # Physics agent observes raw telemetry
        self.observation_spaces = {
            "physics_agent": spaces.Box(low=-np.inf, high=np.inf, shape=(8,)),
            "coach_agent": spaces.Box(low=-np.inf, high=np.inf, shape=(10,))  # +2 for physics output
        }
        
        # Physics outputs trajectory error; Coach outputs haptic
        self.action_spaces = {
            "physics_agent": spaces.Box(low=-10, high=10, shape=(1,)),  # deviation
            "coach_agent": spaces.Box(low=0, high=1, shape=(3,))  # haptic
        }
    
    def step(self, actions):
        # Physics agent calculates error
        trajectory_error = actions["physics_agent"][0]
        
        # Coach decides if rider can handle feedback right now
        rider_stability = self._check_rider_stability()
        
        if rider_stability < 0.5:
            # Rider is correcting a slide - silence haptic to avoid overload
            haptic_action = np.zeros(3)
        else:
            # Rider is stable - deliver coaching
            haptic_action = actions["coach_agent"]
        
        # Apply haptic and update environment
        obs, rewards, dones, infos = self._update_environment(haptic_action)
        
        return obs, rewards, dones, infos
```

### Coordination Strategy

The Coach Agent uses a **gating mechanism**:

```python
def gate_haptic_feedback(physics_error, rider_state):
    """
    Don't overwhelm the rider during critical moments.
    """
    if rider_state['is_correcting_slide']:
        return 0.0  # Silence
    elif rider_state['is_braking_hard']:
        return 0.3 * physics_error  # Gentle hint
    else:
        return 1.0 * physics_error  # Full coaching
```

---

## Phase 4: Offline RL Training

Training an AI on a real motorcycle from scratch is dangerous. We use **Offline Reinforcement Learning**.

### Step 1: Behavior Cloning (BC)

First, the agent learns to **imitate** expert riders:

```python
import minari
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

# Load the offline dataset
dataset = minari.load_dataset("racing_dataset-v0")

# Extract expert demonstrations
expert_obs = dataset.sample_episodes(n=100)['observations']
expert_actions = dataset.sample_episodes(n=100)['actions']

# Clone expert behavior
from imitation.algorithms import bc

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=expert_data,
)

bc_trainer.train(n_epochs=50)
policy = bc_trainer.policy
```

### Step 2: Fine-Tuning with PPO

Once the agent mimics experts, we **fine-tune** in simulation to exceed their performance:

```python
from stable_baselines3 import PPO

# Initialize PPO with the BC-pretrained policy
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1,
)

# Load BC weights
model.policy.load_state_dict(bc_trainer.policy.state_dict())

# Fine-tune in simulation
model.learn(total_timesteps=1_000_000)

# Save final model
model.save("models/ppo_racing_agent.pt")
```

### Multi-Objective Optimization

Using **MO-Gymnasium** to balance multiple objectives:

```python
from mo_gymnasium import MORecordEpisodeStatistics
from mo_gymnasium.utils import MOSyncVectorEnv

# Wrap environment for multi-objective training
env = MORecordEpisodeStatistics(env)

# Train with scalarization (weighted sum)
weights = np.array([1.0, 2.0, 0.5])  # [speed, safety, smooth]

def scalarized_reward(vector_reward):
    return np.dot(vector_reward, weights)
```

---

## Phase 5: Edge AI Deployment

Finally, we deploy the trained AI to the motorcycle.

### Model Quantization

Convert the PyTorch model to 8-bit for microcontrollers:

```python
import torch
import tensorflow as tf

# Export PyTorch model to ONNX
torch.onnx.export(
    model.policy,
    dummy_input,
    "models/policy.onnx",
    export_params=True,
    opset_version=11,
)

# Convert ONNX to TensorFlow Lite
import tf2onnx
tf_model = tf2onnx.convert.from_onnx("models/policy.onnx")

# Quantize to 8-bit integers
converter = tf.lite.TFLiteConverter.from_saved_model("models/tf_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]

tflite_model = converter.convert()

with open("firmware/lib/model.tflite", "wb") as f:
    f.write(tflite_model)
```

### ESP32 Firmware

The C++ firmware reads the BNO055 sensor and executes inference:

```cpp
// firmware/src/main.cpp
#include <Arduino.h>
#include <Adafruit_BNO055.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>

Adafruit_BNO055 bno = Adafruit_BNO055(55);

// TensorFlow Lite Micro setup
const tflite::Model* model;
tflite::MicroInterpreter* interpreter;

void setup() {
  Serial.begin(115200);
  
  // Initialize IMU
  if (!bno.begin()) {
    Serial.println("BNO055 not detected!");
    while (1);
  }
  
  // Load TFLite model
  model = tflite::GetModel(g_model);  // Embedded model bytes
  
  // Create interpreter
  static tflite::MicroMutableOpResolver<4> resolver;
  resolver.AddFullyConnected();
  resolver.AddRelu();
  resolver.AddSoftmax();
  
  static uint8_t tensor_arena[20 * 1024];  // 20KB buffer
  interpreter = new tflite::MicroInterpreter(
    model, resolver, tensor_arena, sizeof(tensor_arena)
  );
  
  interpreter->AllocateTensors();
}

void loop() {
  // Read IMU data
  sensors_event_t event;
  bno.getEvent(&event);
  
  // Prepare input tensor
  float* input = interpreter->input(0)->data.f;
  input[0] = event.acceleration.x;
  input[1] = event.acceleration.y;
  input[2] = event.gyro.z;
  // ... (populate remaining 5 features)
  
  // Run inference (<15ms)
  TfLiteStatus invoke_status = interpreter->Invoke();
  
  // Get haptic output
  float* output = interpreter->output(0)->data.f;
  float left_intensity = output[0];
  float right_intensity = output[1];
  float frequency = output[2];
  
  // Activate haptic motors via PWM
  analogWrite(HAPTIC_LEFT_PIN, left_intensity * 255);
  analogWrite(HAPTIC_RIGHT_PIN, right_intensity * 255);
  
  delay(10);  // 100Hz loop
}
```

### Performance Targets

- **Latency**: <15ms from sensor read to haptic output
- **Sampling Rate**: 100Hz
- **Memory**: <20KB RAM for model inference
- **Power**: <500mW average power consumption

---

## Testing and Validation

### Simulation Testing

```bash
# Test the Gymnasium environment
python -m simulation.test_motorcycle_env

# Train a simple agent
python -m moto_edge_rl.train --env MotorcycleRacing-v0 --algo ppo
```

### Hardware-in-the-Loop (HIL)

```python
# Connect to ESP32 via serial
import serial
ser = serial.Serial('/dev/ttyUSB0', 115200)

# Send test commands and verify haptic response
ser.write(b'TEST_HAPTIC\n')
response = ser.readline()
assert response == b'HAPTIC_OK\n'
```

### Safety Validation

- **Crash Detection**: Verify that friction_usage > 1.0 triggers emergency shutdown
- **Latency Profiling**: Measure end-to-end delay under worst-case scenarios
- **Pilot Feedback**: Conduct controlled tests with professional riders

---

## Conclusion

This implementation guide covers the complete pipeline from data acquisition to edge deployment. The key innovations are:

1. **Offline RL** for safe training without track risk
2. **Multi-agent coordination** for intelligent coaching
3. **Physics-based simulation** with the Kamm Circle model
4. **Edge AI deployment** with <15ms latency

For additional details, refer to the code in `/simulation/motorcycle_env.py` and the firmware in `/firmware/src/`.
