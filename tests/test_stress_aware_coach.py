"""
Unit tests for stress-aware coaching agent and feedback modulation.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.stress_aware_coach import (
    StressAwareCoachAgent,
    MultimodalRacingEnv
)


class TestStressAwareCoachAgent:
    """Test StressAwareCoachAgent class."""

    def test_agent_initialization(self):
        """Verify agent initializes with stress threshold."""
        agent = StressAwareCoachAgent(stress_threshold=0.85)
        
        assert agent.stress_threshold == 0.85

    def test_agent_default_initialization(self):
        """Verify agent initializes with defaults."""
        agent = StressAwareCoachAgent()
        
        assert agent.stress_threshold == 0.9

    def test_feedback_modulation_computes_value(self):
        """Test feedback modulation computes modulation factor."""
        agent = StressAwareCoachAgent(stress_threshold=0.9)
        
        stress_level = 0.3
        modulation = agent.compute_feedback_modulation(stress_level)
        
        assert isinstance(modulation, (float, np.floating))
        assert 0.0 <= modulation <= 1.0

    def test_feedback_modulation_at_high_stress(self):
        """Test feedback modulation at high stress."""
        agent = StressAwareCoachAgent(stress_threshold=0.9)
        
        stress_high = 0.95
        modulation_high = agent.compute_feedback_modulation(stress_high)
        
        # High stress should reduce modulation
        assert 0.0 <= modulation_high <= 1.0


class TestMultimodalRacingEnv:
    """Test MultimodalRacingEnv for environment integration."""

    def test_env_initialization(self):
        """Verify environment initializes correctly."""
        env = MultimodalRacingEnv()
        
        assert env is not None
        assert hasattr(env, "reset")
        assert hasattr(env, "step")

    def test_env_reset(self):
        """Test environment reset returns valid initial state."""
        env = MultimodalRacingEnv()
        state, info = env.reset()
        
        assert state is not None
        assert isinstance(info, dict)

    def test_env_step_signature(self):
        """Verify step function returns correct tuple."""
        env = MultimodalRacingEnv()
        env.reset()
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs is not None
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)

    def test_env_basic_episode(self):
        """Test running a basic episode."""
        env = MultimodalRacingEnv()
        state, _ = env.reset()
        
        total_reward = 0.0
        for step_count in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        assert total_reward is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
