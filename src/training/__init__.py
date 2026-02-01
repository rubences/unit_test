"""
Moto-Edge-RL Training Module

This module provides the hybrid offline-online training pipeline for
motorcycle racing agents using Behavior Cloning and PPO.
"""

from .train_hybrid import (
    BehaviorCloningTrainer,
    PPOTrainer,
    ModelEvaluator,
    save_model,
    main
)

__all__ = [
    'BehaviorCloningTrainer',
    'PPOTrainer',
    'ModelEvaluator',
    'save_model',
    'main'
]
