# Simulation

This directory contains the Gymnasium-based simulation environments for training RL agents.

## Environments

- `motorcycle_env.py` - Custom Gymnasium environment for motorcycle racing dynamics
- `physics/` - Physics models (Kamm Circle, mass transfer, tire friction)
- `tracks/` - Track data and racing line calculations
- `multi_agent/` - PettingZoo multi-agent environments

## Key Features

- **Realistic Physics**: Non-linear motorcycle dynamics including the Kamm Circle (tire friction limits) and weight transfer during braking
- **Multi-Agent Support**: Cooperative agents for trajectory planning and haptic coaching
- **Offline RL**: Integration with Minari for safe offline reinforcement learning
- **Multi-Objective Rewards**: Using MO-Gymnasium for balancing speed, safety, and smooth feedback

## Usage

See the main README and documentation for training examples.
