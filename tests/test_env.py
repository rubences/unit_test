import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.motorcycle_env import MotorcycleEnv


@pytest.fixture
def env():
    env = MotorcycleEnv()
    env.reset(seed=0)
    return env


def test_reset_returns_valid_state():
    env = MotorcycleEnv()
    obs, info = env.reset(seed=42)

    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)
    assert np.isfinite(obs).all()
    assert obs.shape == env.observation_space.shape


def test_action_and_observation_bounds(env):
    raw_action = np.array([2.0, -1.0, 500.0], dtype=np.float32)
    clipped_action = np.clip(raw_action, env.action_space.low, env.action_space.high)

    obs, reward, terminated, truncated, info = env.step(raw_action)

    assert env.action_space.contains(env.last_action)
    assert np.allclose(env.last_action, clipped_action)
    assert env.observation_space.contains(obs)
    assert np.isfinite(obs).all()
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_reward_is_finite(env):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert not np.isnan(reward)
    assert np.isfinite(reward)
    assert env.observation_space.contains(obs)
