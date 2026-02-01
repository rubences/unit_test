"""
Pipeline Configuration and Hyperparameters

This file documents all configurable parameters for the Moto-Edge-RL
training and deployment pipeline.
"""

# ============================================================================
# DATA GENERATION CONFIGURATION
# ============================================================================

DATA_CONFIG = {
    # Synthetic data generation
    "data_generation": {
        "num_laps_pro": 100,           # Professional rider laps
        "num_laps_amateur": 100,       # Amateur rider laps
        "lap_length": 300,             # Timesteps per lap (5 seconds @ 60Hz)
        "output_format": "hdf5",       # Minari format
        "seed": 42,                    # Reproducibility
        
        # Pro rider characteristics
        "pro_rider": {
            "brake_intensity": 0.9,    # Strong braking
            "throttle_smoothness": 0.95,
            "lean_aggressiveness": 0.85,
            "racing_line_accuracy": 0.95,  # High consistency
        },
        
        # Amateur rider characteristics
        "amateur_rider": {
            "brake_intensity": 0.6,    # Variable braking
            "throttle_smoothness": 0.6,
            "lean_aggressiveness": 0.6,
            "racing_line_accuracy": 0.7,  # Lower consistency
            "input_noise": 0.15,       # Jerkiness
        }
    },
    
    # Environment simulation
    "environment": {
        "max_velocity": 100.0,         # m/s (~360 km/h)
        "max_lean_angle": 60.0,        # degrees
        "max_lateral_g": 2.0,          # G-forces
        "tire_friction_coeff": 1.5,    # Dry tarmac
        "gravity": 9.81,               # m/s^2
    },
    
    # Observation and action spaces
    "spaces": {
        "observation_dim": 8,
        "action_dim": 3,
        "observation_names": [
            "velocity",
            "roll_angle",
            "lateral_g",
            "distance_to_apex",
            "throttle_position",
            "brake_pressure",
            "racing_line_deviation",
            "tire_friction_usage"
        ],
        "action_names": [
            "haptic_left_intensity",
            "haptic_right_intensity",
            "haptic_frequency"
        ]
    }
}


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

TRAINING_CONFIG = {
    # Behavior Cloning (Offline Pre-training)
    "behavior_cloning": {
        "enabled": True,
        "epochs": 5,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "data_split_ratio": {
            "pro": 0.7,                # 70% professional rider
            "amateur": 0.3,            # 30% amateur rider
        },
        "optimizer": "adam",
        "loss_function": "mse",
        "normalize_observations": True,
    },
    
    # PPO Fine-tuning (Online Training)
    "ppo": {
        "total_timesteps": 100000,     # 100K timesteps
        "learning_rate": 3e-4,
        "n_steps": 2048,               # Rollout length
        "batch_size": 64,              # Mini-batch size
        "n_epochs": 10,                # Updates per cycle
        "gamma": 0.99,                 # Discount factor
        "gae_lambda": 0.95,            # GAE parameter
        "clip_range": 0.2,             # PPO clip coefficient
        "clip_range_vf": None,         # No VF clipping
        "normalize_advantage": True,
        "ent_coef": 0.0,               # Entropy regularization
        "vf_coef": 0.5,                # Value function weight
        "max_grad_norm": 0.5,
        "use_sde": False,              # State-dependent exploration
        "policy": "MlpPolicy",         # Network architecture
        "policy_kwargs": {
            "net_arch": [256, 256],    # Hidden layer sizes
            "activation_fn": "relu",
        },
        "device": "cpu",               # Edge compatibility
        "verbose": 1,
    },
    
    # Reward function
    "reward": {
        "lap_time_weight": -1.0,       # Penalty for slow laps
        "safety_bonus": 0.1,           # Bonus for safe driving
        "lateral_g_threshold": 1.8,    # Safe G-force limit
        "tire_friction_threshold": 0.95,  # Safe friction limit
        "action_regularization": 0.01, # Smooth control penalty
    },
    
    # Callbacks and checkpointing
    "callbacks": {
        "checkpoint_frequency": 5000,  # Save every 5K steps
        "eval_frequency": 5000,        # Evaluate every 5K steps
        "eval_episodes": 10,
        "tensorboard_logging": True,
    }
}


# ============================================================================
# DEPLOYMENT CONFIGURATION
# ============================================================================

DEPLOYMENT_CONFIG = {
    # Model export pipeline
    "export": {
        "formats": ["onnx", "tensorflow", "tflite"],
        "opset_version": 14,           # ONNX opset
        "tensorflow_version": "2.13+",
    },
    
    # TFLite conversion
    "tflite": {
        "target_ops": ["tflite_builtins"],
        "optimizations": ["default"],
        "inference_type": "float32",   # Inference precision
    },
    
    # Quantization (int8)
    "quantization": {
        "enabled": True,
        "method": "dynamic_range",     # dynamic_range or full_integer
        "weight_bits": 8,
        "activation_bits": 8,
        "representative_data_size": 100,  # For full quantization
        "target_size_reduction": 4.0,  # 4x smaller
    },
    
    # ESP32 Target Specifications
    "esp32": {
        "cpu": "Xtensa LX6",
        "frequency": 240,              # MHz
        "cores": 2,
        "sram": 520,                   # KB (shared)
        "flash": 4096,                 # KB (shared)
        "model_size_max": 2048,        # KB
        "inference_latency_target": 50,  # ms
        "inference_frequency": 20,     # Hz (haptic update rate)
    },
    
    # Validation
    "validation": {
        "test_inference": True,
        "check_output_shape": True,
        "compare_with_original": True,
        "accuracy_threshold": 0.95,    # 95% similarity to original
    }
}


# ============================================================================
# PIPELINE EXECUTION CONFIGURATION
# ============================================================================

PIPELINE_CONFIG = {
    "steps": {
        "data_generation": {
            "enabled": True,
            "skip_if_exists": False,   # Regenerate if exists
        },
        "training": {
            "enabled": True,
            "skip_if_exists": False,
        },
        "deployment": {
            "enabled": True,
            "skip_if_exists": False,
        }
    },
    
    # Paths
    "paths": {
        "data_dir": "data/processed",
        "model_dir": "models",
        "deployment_dir": "models/edge_deployment",
        "log_dir": "logs",
        "checkpoint_dir": "models/checkpoints",
    },
    
    # Logging and monitoring
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file_logging": True,
        "tensorboard": True,
        "wandb": False,  # Weights & Biases integration
    },
    
    # Performance
    "performance": {
        "num_parallel_envs": 4,        # Parallel environments for rollouts
        "batch_processing": True,
        "device": "cpu",               # 'cpu' or 'cuda' (edge-friendly)
    },
    
    # Random seed
    "seed": 42,
}


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

CONFIGS = {
    "quick_test": {
        "data_laps": 10,
        "ppo_timesteps": 10000,
        "eval_episodes": 5,
        "description": "Quick test (30 minutes)"
    },
    "small": {
        "data_laps": 50,
        "ppo_timesteps": 50000,
        "eval_episodes": 5,
        "description": "Small training (2-3 hours)"
    },
    "standard": {
        "data_laps": 100,
        "ppo_timesteps": 100000,
        "eval_episodes": 10,
        "description": "Standard training (6-8 hours)"
    },
    "large": {
        "data_laps": 200,
        "ppo_timesteps": 200000,
        "eval_episodes": 20,
        "description": "Large training (12-14 hours)"
    },
    "xlarge": {
        "data_laps": 500,
        "ppo_timesteps": 500000,
        "eval_episodes": 20,
        "description": "Extra large training (24+ hours)"
    }
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_config(preset_name: str = "standard") -> dict:
    """
    Get configuration preset.
    
    Args:
        preset_name: Configuration preset name
        
    Returns:
        Configuration dictionary
    """
    if preset_name not in CONFIGS:
        raise ValueError(f"Unknown preset: {preset_name}")
    return CONFIGS[preset_name]


def print_config():
    """Print current configuration."""
    import json
    print("\nMOTO-EDGE-RL PIPELINE CONFIGURATION")
    print("=" * 70)
    print("\nDATA CONFIGURATION:")
    print(json.dumps(DATA_CONFIG, indent=2))
    print("\nTRAINING CONFIGURATION:")
    print(json.dumps(TRAINING_CONFIG, indent=2))
    print("\nDEPLOYMENT CONFIGURATION:")
    print(json.dumps(DEPLOYMENT_CONFIG, indent=2))
    print("\nPIPELINE CONFIGURATION:")
    print(json.dumps(PIPELINE_CONFIG, indent=2))


if __name__ == "__main__":
    print_config()
